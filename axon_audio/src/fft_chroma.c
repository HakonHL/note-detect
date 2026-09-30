#include "fft_chroma.h"

#include <arm_math.h>
#include <math.h>
#include <string.h>

#define RATE 16000
#define MIN_HZ 65.0f
#define MAX_HZ 2093.0f
#define INPUT_ZERO (-128)
#define QUANT_GAIN_NUM 1000000ULL
#define QUANT_GAIN_DEN 3683ULL

static arm_rfft_instance_q15 fft;
static q15_t ring[CHROMA_FFT_SIZE];
static q15_t window[CHROMA_FFT_SIZE];
static q15_t fft_input[CHROMA_FFT_SIZE];
static q15_t fft_output[2 * CHROMA_FFT_SIZE];
static int8_t bin_pitch[CHROMA_FFT_SIZE / 2 + 1];
static int8_t history[CHROMA_FRAMES][CHROMA_CLASSES];
static uint32_t squared_history[CHROMA_FRAMES];
static unsigned int write_at;
static unsigned int sample_count;
static unsigned int hop_count;
static unsigned int frame_count;

static uint64_t sqrt_u64(uint64_t value)
{
	uint64_t root = 0;
	uint64_t bit = 1ULL << 62;

	while (bit > value) {
		bit >>= 2;
	}
	while (bit) {
		if (value >= root + bit) {
			value -= root + bit;
			root = (root >> 1) + bit;
		} else {
			root >>= 1;
		}
		bit >>= 2;
	}
	return root;
}

int fft_chroma_init(void)
{
	if (arm_rfft_init_q15(&fft, CHROMA_FFT_SIZE, 0, 1) != ARM_MATH_SUCCESS) {
		return -1;
	}

	for (unsigned int i = 0; i < CHROMA_FFT_SIZE; ++i) {
		float hann = 0.5f * (1.0f - cosf(2.0f * PI * i / CHROMA_FFT_SIZE));

		window[i] = (q15_t)lroundf(32767.0f * hann);
	}
	for (unsigned int i = 0; i <= CHROMA_FFT_SIZE / 2; ++i) {
		float hz = (float)i * RATE / CHROMA_FFT_SIZE;

		bin_pitch[i] = -1;
		if (hz < MIN_HZ || hz > MAX_HZ) {
			continue;
		}
		float pitch = 69.0f + 12.0f * log2f(hz / 440.0f);
		int nearest = (int)lroundf(pitch);

		if (fabsf(pitch - nearest) < 0.5f) {
			bin_pitch[i] = (int8_t)((nearest % 12 + 12) % 12);
		}
	}
	memset(ring, 0, sizeof(ring));
	memset(history, INPUT_ZERO, sizeof(history));
	write_at = sample_count = hop_count = frame_count = 0;
	return 0;
}

static void calculate_frame(void)
{
	uint32_t energies[CHROMA_CLASSES] = {0};
	uint64_t square_sum = 0;
	unsigned int slot = frame_count % CHROMA_FRAMES;

	for (unsigned int i = 0; i < CHROMA_FFT_SIZE; ++i) {
		int32_t v = ring[(write_at + i) % CHROMA_FFT_SIZE];

		square_sum += (int64_t)v * v;
		fft_input[i] = (q15_t)((v * window[i]) >> 15);
	}
	arm_rfft_q15(&fft, fft_input, fft_output);
	for (unsigned int i = 0; i <= CHROMA_FFT_SIZE / 2; ++i) {
		int pc = bin_pitch[i];

		if (pc < 0) {
			continue;
		}
		int32_t real = fft_output[2 * i];
		int32_t imag = fft_output[2 * i + 1];

		energies[pc] += (uint32_t)sqrt_u64((uint64_t)(real * real + imag * imag));
	}

	uint64_t norm2 = 0;

	for (unsigned int p = 0; p < CHROMA_CLASSES; ++p) {
		norm2 += (uint64_t)energies[p] * energies[p];
	}
	uint64_t denom = sqrt_u64(norm2) * QUANT_GAIN_DEN;

	for (unsigned int p = 0; p < CHROMA_CLASSES; ++p) {
		int value = INPUT_ZERO;

		if (denom != 0) {
			value += (int)(((uint64_t)energies[p] * QUANT_GAIN_NUM + denom / 2) /
				       denom);
		}

		history[slot][p] = (int8_t)(value < -128 ? -128 : value > 127 ? 127 : value);
	}
	squared_history[slot] = (uint32_t)(square_sum / CHROMA_FFT_SIZE);
	++frame_count;
}

bool fft_chroma_feed(const int16_t *audio, unsigned int count)
{
	bool ready = false;

	for (unsigned int i = 0; i < count; ++i) {
		ring[write_at] = audio[i];
		write_at = (write_at + 1) % CHROMA_FFT_SIZE;
		++sample_count;
		if (++hop_count == CHROMA_HOP) {
			hop_count = 0;
			if (sample_count >= CHROMA_FFT_SIZE) {
				calculate_frame();
				ready = true;
			}
		}
	}
	return ready;
}

bool fft_chroma_features(int8_t output[CHROMA_FRAMES][CHROMA_CLASSES],
			 uint32_t *rms_squared)
{
	if (frame_count < CHROMA_FRAMES) {
		return false;
	}
	for (unsigned int i = 0; i < CHROMA_FRAMES; ++i) {
		unsigned int slot = (frame_count + i) % CHROMA_FRAMES;

		memcpy(output[i], history[slot], CHROMA_CLASSES);
	}
	*rms_squared = squared_history[(frame_count - 1) % CHROMA_FRAMES];
	return true;
}
