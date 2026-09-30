"""Reference for the embedded 4096-point FFT pitch-class frontend.

Unlike the training CQT, this computes linear-frequency FFT magnitudes folded
into 12 pitch classes. Use compare_frontend.py to quantify the domain shift.
"""

import numpy as np
from scipy.signal import get_window

RATE = 16000
FFT_SIZE = 4096
HOP = 320
MIN_HZ = 65.0
MAX_HZ = 2093.0


def bin_mapping():
    hz = np.arange(FFT_SIZE // 2 + 1) * RATE / FFT_SIZE
    pitch = 69 + 12 * np.log2(np.maximum(hz, 1.0) / 440.0)
    nearest = np.rint(pitch).astype(int)
    valid = (hz >= MIN_HZ) & (hz <= MAX_HZ) & (np.abs(pitch - nearest) < 0.5)
    return np.where(valid, nearest % 12, -1)


BINS = bin_mapping()
WINDOW = get_window('hann', FFT_SIZE, fftbins=True).astype(np.float32)


def chroma_frames(samples, frames=32):
    """Return the first centered frames, zero-padding the input at each end."""
    samples = np.asarray(samples, dtype=np.float32)
    padded = np.pad(samples, (FFT_SIZE // 2, FFT_SIZE // 2))
    result = np.zeros((frames, 12, 1), dtype=np.float32)
    for i in range(frames):
        start = i * HOP
        block = padded[start:start + FFT_SIZE]
        if len(block) < FFT_SIZE:
            block = np.pad(block, (0, FFT_SIZE - len(block)))
        magnitude = np.abs(np.fft.rfft(block * WINDOW))
        for pc in range(12):
            result[i, pc, 0] = magnitude[BINS == pc].sum()
        norm = np.linalg.norm(result[i, :, 0])
        if norm:
            result[i, :, 0] /= norm
    return result
