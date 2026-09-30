#ifndef FFT_CHROMA_H_
#define FFT_CHROMA_H_

#include <stdbool.h>
#include <stdint.h>

#define CHROMA_FFT_SIZE 4096
#define CHROMA_HOP 320
#define CHROMA_FRAMES 32
#define CHROMA_CLASSES 12

int fft_chroma_init(void);
/* Feed 16 kHz mono signed PCM. Returns true whenever a new 20 ms frame is ready. */
bool fft_chroma_feed(const int16_t *audio, unsigned int count);
/* Copy chronological [32][12] quantized features; valid after 32 frames. */
bool fft_chroma_features(int8_t output[CHROMA_FRAMES][CHROMA_CLASSES],
			 uint32_t *rms_squared);

#endif
