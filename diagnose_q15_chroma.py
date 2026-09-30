#!/usr/bin/env python3
"""Compare float chroma with CMSIS Q15 using the MCU's integer arithmetic.

No ADC/FIR simulation: the supplied audio is assumed to be post-decimation.
CMSIS host wrapper arithmetic is a reference, not proven target byte parity.
"""
from collections import Counter
import json
import math

import cmsisdsp as dsp
import numpy as np
import soundfile as sf

from analyze_real_chords import ROOT, LABELS, Model, names, fft_stream
from fft_chroma import BINS, FFT_SIZE, HOP, RATE, WINDOW


def q15_stream(audio, normalize=False):
    instance = dsp.arm_rfft_instance_q15()
    if dsp.arm_rfft_init_q15(instance, FFT_SIZE, 0, 1) != 0:
        raise RuntimeError('Q15 FFT initialization failed')
    window = np.floor(WINDOW * 32767 + .5).astype(np.int64)
    signal = np.clip(np.rint(audio * 32768), -32768, 32767).astype(np.int64)
    ends = np.arange(((FFT_SIZE + HOP - 1) // HOP) * HOP, len(audio) + 1, HOP)
    features = []
    for end in ends:
        raw = signal[end-FFT_SIZE:end].copy()
        if normalize:
            # Experimental power-of-two gain before the FFT; preserve raw RMS
            # separately in firmware so this never raises the silence gate.
            peak = int(np.max(np.abs(raw)))
            shift = 0
            while peak > 0 and peak <= 8191 and shift < 14:
                peak *= 2
                shift += 1
            raw <<= shift
        block = ((raw * window) >> 15).astype(np.int16)
        output = dsp.arm_rfft_q15(instance, block).astype(np.int64)
        pairs = output[:FFT_SIZE + 2].reshape(-1, 2)
        magnitude = np.array([math.isqrt(int(r*r + i*i)) for r, i in pairs])
        energy = np.array([magnitude[BINS == pc].sum() for pc in range(12)])
        denominator = math.isqrt(int(np.sum(energy ** 2))) * 3683
        q = np.full(12, -128, dtype=np.int64)
        if denominator:
            q += (energy * 1000000 + denominator // 2) // denominator
        # Dequantize here; Model requantizes, recovering the same int8 tensor.
        features.append((np.clip(q, -128, 127) + 128) * .0036825458519160748)
    return np.array(features)


def main():
    directory = ROOT / 'captures/real_chords_baseline'
    model = Model(ROOT / 'artifacts/note_cnn_fft')
    results = []
    for path in sorted(directory.glob('*.wav')):
        if '_peak-' in path.stem:
            continue
        chord = next(key for key in LABELS if path.stem.startswith(key))
        audio, sr = sf.read(path, dtype='float32')
        assert sr == RATE
        ends, reference, rms = fft_stream(audio)
        active = rms >= max(10 ** (-65/20), rms.max() * 10 ** (-25/20))
        selected = [i for i in range(31, len(ends), 10)
                    if active[i] and np.mean(active[i-31:i+1]) >= .75]
        normalized = audio * (10 ** (-3/20) / max(np.max(np.abs(audio)), 1e-12))
        sf.write(directory / (path.stem + '_peak-3.wav'), normalized, RATE, subtype='PCM_24')
        experiments = [(gain, False) for gain in [1.0, .1, .03, .01, .003]]
        experiments += [(gain, True) for gain in [1.0, .1, .01]]
        for gain, normalized_fft in experiments:
            fixed = q15_stream(audio * gain, normalize=normalized_fft)
            predicted = [names(np.flatnonzero(model(fixed[i-31:i+1]) >= model.threshold)) for i in selected]
            cosine = np.sum(reference * fixed, axis=1) / (np.linalg.norm(fixed, axis=1) + 1e-12)
            item = {'file': path.name, 'input_gain': gain,
                    'experimental_pre_fft_gain': normalized_fft,
                    'maximum_q15_rms': float(rms.max() * gain * 32768),
                    'mean_active_feature_cosine': float(np.mean(cosine[active])),
                    'exact_set_rate': sum(p == names(LABELS[chord]) for p in predicted) / max(len(predicted), 1),
                    'E_detection_rate': sum('E' in p.split() for p in predicted) / max(len(predicted), 1),
                    'common_sets': Counter(predicted).most_common(4)}
            results.append(item)
            print(f"{path.stem:14} gain={gain:.3f} preFFTgain={normalized_fft} RMSmax={item['maximum_q15_rms']:7.1f} "
                  f"cos={item['mean_active_feature_cosine']:.3f} exact={item['exact_set_rate']:.1%} "
                  f"sets={item['common_sets']}", flush=True)
    (directory / 'q15_gain_sweep.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()