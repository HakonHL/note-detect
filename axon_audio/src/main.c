/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>
#include <math.h>

#include "fft_chroma.h"
#include "i2s_audio.h"
#include "generated/nrf_axon_model_guitar_note_cnn_.h"
#include "generated/nrf_axon_model_guitar_note_cnn_test_vectors_.h"
LOG_MODULE_REGISTER(guitar_note_test);

#define OUTPUT_SIZE (12)
#define LIVE_REPORT_INTERVAL_MS 200

/* From artifacts/note_cnn_fft/postprocess.json: sigmoid(logit) >= 0.45. */
#define NOTE_LOGIT_THRESHOLD (-0.20067f)

static const char *const note_names[OUTPUT_SIZE] = {
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
};

static void format_notes(const int32_t *logits, int32_t threshold, char *buf, size_t len)
{
	size_t used = 0;

	buf[0] = '\0';
	for (size_t i = 0; i < OUTPUT_SIZE; i++) {
		if (logits[i] >= threshold) {
			used += snprintk(buf + used, len - used, "%s%s", used ? " " : "",
					 note_names[i]);
		}
	}
	if (used == 0) {
		snprintk(buf, len, "none");
	}
}

static void sync_flow(void)
{
	nrf_axon_result_e result;
	/* Raw int32 outputs are logits scaled by 2^output_dequant_round. */
	const int32_t threshold = (int32_t)(NOTE_LOGIT_THRESHOLD *
					    (float)(1 << model_guitar_note_cnn.output_dequant_round));

	result = nrf_axon_nn_model_validate(&model_guitar_note_cnn);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Model validation failed, err %d", result);
		return;
	}

	LOG_INF("Validated compiled model; running %u test vectors, raw threshold %d",
		ARRAY_SIZE(guitar_note_cnn_input_test_vectors), threshold);

	for (size_t vector = 0; vector < ARRAY_SIZE(guitar_note_cnn_input_test_vectors); vector++) {
		const int8_t *input = (const int8_t *)guitar_note_cnn_input_test_vectors[vector];
		const int32_t *expected = guitar_note_cnn_expected_output_vectors[vector];
		int32_t output[OUTPUT_SIZE];
		char detected[48];
		char reference[48];
		int32_t max_abs_diff = 0;
		uint32_t start = k_cycle_get_32();

		result = nrf_axon_nn_model_infer_sync(&model_guitar_note_cnn, input,
						       (int8_t *)output);
		uint32_t cycles = k_cycle_get_32() - start;

		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("Inference failed for vector %u, err %d", vector, result);
			return;
		}

		for (size_t i = 0; i < OUTPUT_SIZE; i++) {
			const int32_t diff = output[i] - expected[i];

			max_abs_diff = MAX(max_abs_diff, diff < 0 ? -diff : diff);
		}
		format_notes(output, threshold, detected, sizeof(detected));
		format_notes(expected, threshold, reference, sizeof(reference));

		LOG_INF("vector %u: notes [%s], reference [%s], max abs diff %d, %u us", vector,
			detected, reference, max_abs_diff,
			k_cyc_to_us_floor32(cycles));
	}
}

static int8_t live_features[CHROMA_FRAMES][CHROMA_CLASSES];

static void live_flow(void)
{
	const int32_t threshold = (int32_t)(NOTE_LOGIT_THRESHOLD *
					    (float)(1 << model_guitar_note_cnn.output_dequant_round));
	int err = fft_chroma_init();

	if (err) {
		LOG_ERR("FFT init failed: %d", err);
		return;
	}
	err = i2s_audio_init();
	if (err) {
		LOG_ERR("I2S init failed: %d", err);
		return;
	}
	LOG_INF("Live PCM1808 -> FIR /6 -> 4096-point FFT chroma -> Axon started");
	uint32_t started_ms = k_uptime_get_32();
	uint32_t next_report_ms = started_ms + LIVE_REPORT_INTERVAL_MS;

	for (;;) {
		void *block;
		size_t size;

		err = i2s_audio_read(&block, &size, 1000);
		if (err) {
			LOG_ERR("I2S read failed: %d", err);
			return;
		}
		bool ready = fft_chroma_feed(block, size / sizeof(int16_t));

		free_i2s_audio_buffer(block);
		if (!ready) {
			continue;
		}
		uint32_t rms_squared;

		if (!fft_chroma_features(live_features, &rms_squared)) {
			continue;
		}
		int32_t logits[OUTPUT_SIZE];
		nrf_axon_result_e result = nrf_axon_nn_model_infer_sync(
			&model_guitar_note_cnn, (const int8_t *)live_features, (int8_t *)logits);
		if (result != NRF_AXON_RESULT_SUCCESS) {
			LOG_ERR("Live inference failed: %d", result);
			return;
		}
		char notes[48];

		if (rms_squared < 340) {
			snprintk(notes, sizeof(notes), "none");
		} else {
			format_notes(logits, threshold, notes, sizeof(notes));
		}
		uint32_t now_ms = k_uptime_get_32();

		if ((int32_t)(now_ms - next_report_ms) >= 0) {
			LOG_INF("live t=%u ms: [%s] rms=%u", now_ms - started_ms, notes,
				(uint32_t)sqrtf((float)rms_squared));
			/* Keep the cadence tied to uptime rather than accumulating logging delays. */
			do {
				next_report_ms += LIVE_REPORT_INTERVAL_MS;
			} while ((int32_t)(now_ms - next_report_ms) >= 0);
		}
	}
}

int main(void)
{
	nrf_axon_result_e result;
	int err;

	LOG_INF("Hello Axon sample");
	LOG_INF("Initializing Axon NPU");

	__ASSERT(model_guitar_note_cnn.inputs[model_guitar_note_cnn.external_input_ndx]
				 .dimensions.byte_width == 1,
		 "Model input data type different than expected");
	__ASSERT(model_guitar_note_cnn.output_dimensions.byte_width == 4,
		 "Model output data type different than expected");

	result = nrf_axon_platform_init();
	if (result != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Axon NPU platform initialization failed, err %d", result);
		return -1;
	}

	/* This model has no persistent variables, so this initialization is a no-op. */
	err = nrf_axon_nn_model_init_vars(&model_guitar_note_cnn);
	if (err) {
		LOG_ERR("Model persistent variables initialization failed, err %d", err);
		return -1;
	}

	sync_flow();
	live_flow();

	return 0;
}
