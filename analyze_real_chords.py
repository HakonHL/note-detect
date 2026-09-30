#!/usr/bin/env python3
"""Non-destructive labelled-recording baseline; no training or firmware changes.

FFT windows follow MCU frame timing, but use float FFT, not CMSIS Q15.
CQT is an offline, centered comparison and is not an embedded implementation.
Filename chord labels describe intended notes, not verified individual strings.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

import librosa
import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter
from scipy.signal import resample_poly

from fft_chroma import BINS, FFT_SIZE, HOP, RATE, WINDOW

ROOT = Path(__file__).resolve().parent
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
LABELS = {'asus2': {9, 11, 4}, 'dsus2': {2, 4, 9},
          'dsus4': {2, 7, 9}, 'fmaj7': {5, 9, 0, 4}}


def names(notes):
    return ' '.join(NAMES[i] for i in sorted(notes)) or 'none'


class Model:
    def __init__(self, directory):
        self.net = Interpreter(model_path=str(directory / 'note_cnn_int8.tflite'))
        self.net.allocate_tensors()
        self.input = self.net.get_input_details()[0]
        self.output = self.net.get_output_details()[0]
        self.threshold = json.loads((directory / 'postprocess.json').read_text())['logit_threshold']

    def __call__(self, feature):
        scale, zero = self.input['quantization']
        value = np.clip(np.rint(feature / scale + zero), -128, 127).astype(np.int8)
        self.net.set_tensor(self.input['index'], value[None, ..., None])
        self.net.invoke()
        scale, zero = self.output['quantization']
        return (self.net.get_tensor(self.output['index'])[0].astype(float) - zero) * scale


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not supported')
            if not target.exists():
                source.extract(entry, destination)


def fft_stream(audio):
    # MCU first frame ends at 4160 samples (first hop after the FFT fills).
    ends = np.arange(((FFT_SIZE + HOP - 1) // HOP) * HOP, len(audio) + 1, HOP)
    features = []
    rms = []
    for end in ends:
        block = audio[end - FFT_SIZE:end]
        magnitude = np.abs(np.fft.rfft(block * WINDOW))
        chroma = np.array([magnitude[BINS == pc].sum() for pc in range(12)])
        features.append(chroma / max(np.linalg.norm(chroma), 1e-12))
        rms.append(np.sqrt(np.mean(block ** 2)))
    return ends, np.asarray(features), np.asarray(rms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'chords.zip')
    parser.add_argument('--out', type=Path, default=ROOT / 'captures/real_chords_baseline')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recordings = ROOT / 'recordings/real_chords'
    extract(args.archive, recordings)
    models = {'fft': Model(ROOT / 'artifacts/note_cnn_fft'),
              'cqt': Model(ROOT / 'artifacts/note_cnn_pitch')}
    summary = []
    for path in sorted(recordings.rglob('*.mp3')):
        chord = next((key for key in LABELS if path.stem.startswith(key)), None)
        if chord is None:
            raise ValueError(f'Unrecognized label: {path.name}')
        truth = LABELS[chord]
        original, sr = sf.read(path, dtype='float32', always_2d=True)
        mono = original.mean(axis=1)
        gcd = np.gcd(sr, RATE)
        audio = resample_poly(mono, RATE // gcd, sr // gcd).astype(np.float32)
        wav = args.out / (path.stem + '.wav')
        sf.write(wav, audio, RATE, subtype='PCM_24')
        ends, fft, rms = fft_stream(audio)
        cqt = librosa.feature.chroma_cqt(y=audio, sr=RATE, hop_length=HOP,
            fmin=librosa.note_to_hz('C1'), bins_per_octave=36, n_octaves=7,
            tuning=0.0, norm=2).T
        # Compare the same physical FFT centers; CQT uses future context offline.
        centers = np.rint((ends - FFT_SIZE / 2) / HOP).astype(int)
        cqt = cqt[centers]
        active = rms >= max(10 ** (-65 / 20), float(rms.max()) * 10 ** (-25 / 20))
        rows = []
        for i in range(31, len(ends), 10):
            # Reject attack/quiet transitions from summary scores, retaining them in CSV.
            scored = bool(active[i] and np.mean(active[i-31:i+1]) >= 0.75)
            row = {'time_s': ends[i] / RATE, 'rms_dbfs': 20 * np.log10(rms[i] + 1e-12),
                   'scored': scored, 'truth': names(truth)}
            for key, feature in [('fft', fft), ('cqt', cqt)]:
                logits = models[key](feature[i-31:i+1])
                row[key + '_notes'] = names(np.flatnonzero(logits >= models[key].threshold))
                for pc, note in enumerate(NAMES):
                    row[key + '_logit_' + note] = float(logits[pc])
            for pc, note in enumerate(NAMES):
                row['fft_energy_' + note] = float(fft[i-31:i+1, pc].mean())
            rows.append(row)
        with (args.out / (path.stem + '.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        scored_rows = [r for r in rows if r['scored']]
        tuning = float(librosa.estimate_tuning(y=audio, sr=RATE)) * 100
        item = {'file': path.name, 'truth': names(truth), 'duration_s': len(mono) / sr,
                'sample_rate': sr, 'channels': original.shape[1],
                'peak_dbfs': float(20 * np.log10(np.max(np.abs(original)) + 1e-12)),
                'samples_ge_0999_fraction': float(np.mean(np.abs(original) >= .999)),
                'spectral_tuning_cents': tuning, 'scored_windows': len(scored_rows),
                'active_gate_dbfs': float(20 * np.log10(max(10 ** (-65/20), rms.max() * 10 ** (-25/20)))),
                'models': {}}
        for key in models:
            selected = [set(r[key + '_notes'].split()) - {'none'} for r in scored_rows]
            intended = set(names(truth).split())
            tp = sum(len(s & intended) for s in selected)
            fp = sum(len(s - intended) for s in selected)
            fn = sum(len(intended - s) for s in selected)
            item['models'][key] = {
                'exact_set_rate': sum(s == intended for s in selected) / max(len(selected), 1),
                'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                'common_sets': Counter(r[key + '_notes'] for r in scored_rows).most_common(5),
                'note_detection_rate': {note: sum(note in s for s in selected) / max(len(selected), 1) for note in NAMES},
                'median_logits': {note: float(np.median([r[key + '_logit_' + note] for r in scored_rows])) for note in NAMES}}
        summary.append(item)
        print(f"{path.name:20} {item['duration_s']:5.1f}s truth=[{names(truth)}] "
              f"peak={item['peak_dbfs']:.1f} dBFS tuning~{tuning:+.0f}c "
              f"FFT={item['models']['fft']['exact_set_rate']:.1%} "
              f"CQT={item['models']['cqt']['exact_set_rate']:.1%}", flush=True)
        print('  FFT sets:', item['models']['fft']['common_sets'], flush=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Results: {args.out}')


if __name__ == '__main__':
    main()