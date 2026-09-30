#!/usr/bin/env python3
"""Measure the FFT-chroma versus librosa-CQT feature and model-output gap."""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from eval_capture import Model, NOTE_NAMES, align, dataset_entry, load_mono
from fft_chroma import chroma_frames
from train_chord_cnn import chroma_features

warnings.filterwarnings('ignore', message='n_fft=.*is too large')
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset_notes')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts/note_cnn_pitch')
    parser.add_argument('--fft-model-dir', type=Path, default=ROOT / 'artifacts/note_cnn_fft')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--capture', type=Path,
                        default=ROOT / 'captures/pcm1808-heldout.raw24.wav')
    args = parser.parse_args()
    model = Model(args.model_dir / 'note_cnn_int8.tflite')
    threshold = json.loads((args.model_dir / 'postprocess.json').read_text())['logit_threshold']
    fft_model = Model(args.fft_model_dir / 'note_cnn_int8.tflite')
    fft_threshold = json.loads((args.fft_model_dir / 'postprocess.json').read_text())['logit_threshold']
    with (args.dataset / 'labels.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(91)
    selected = [rows[i] for i in rng.choice(len(rows), min(args.count, len(rows)), replace=False)]
    scores, fft_hits, cqt_hits = [], [], []
    for row in selected:
        audio, sr = sf.read(args.dataset / row['path'], dtype='float32', always_2d=True)
        if sr != 16000:
            audio = resample_poly(audio, 16000, sr)
        mono = audio.mean(axis=1)
        fft = chroma_frames(mono)
        cqt = chroma_features(mono, 16000, model.frames)
        cosine = np.sum(fft * cqt, axis=1) / (np.linalg.norm(fft, axis=1) * np.linalg.norm(cqt, axis=1) + 1e-9)
        truth = {int(note) % 12 for note in row['midi_notes'].split()}
        predicted = lambda net, feature, cutoff: set(np.flatnonzero(net(feature) >= cutoff))
        scores.append(np.mean(cosine))
        fft_hits.append(predicted(fft_model, fft, fft_threshold) == truth)
        cqt_hits.append(predicted(model, cqt, threshold) == truth)
    print(f'{len(selected)} clips: feature cosine mean {np.mean(scores):.3f}, median {np.median(scores):.3f}')
    print(f'Exact notes with matched models: librosa CQT {np.mean(cqt_hits):.1%}, '
          f'4096 FFT {np.mean(fft_hits):.1%}')

    segments_path = args.capture.with_name(args.capture.name.split('.')[0] + '.segments.csv')
    recording = load_mono(args.capture, 16000)
    with segments_path.open(newline='') as stream:
        segments = list(csv.DictReader(stream))
    print('\nPCM1808 capture: aligned source -> frontend -> note model')
    print('clip                  truth          CQT model      FFT model       feature cos')
    for segment in segments:
        source_path = Path(segment['file'])
        _, row = dataset_entry(source_path)
        if row is None:
            continue
        source = load_mono(source_path, 16000)
        offset, _ = align(recording, source, float(segment['start_s']), 16000, .3, 1.0)
        if offset is None or offset + len(source) > len(recording):
            continue
        audio = recording[offset:offset + len(source)]
        cqt = chroma_features(audio.astype(np.float32), 16000, 32)
        fft = chroma_frames(audio)
        truth = {int(note) % 12 for note in row['midi_notes'].split()}
        text = lambda notes: ' '.join(NOTE_NAMES[n] for n in sorted(notes)) or 'none'
        notes = lambda net, feature, cutoff: set(np.flatnonzero(net(feature) >= cutoff))
        cosine = np.mean(np.sum(fft * cqt, axis=1) /
                         (np.linalg.norm(fft, axis=1) * np.linalg.norm(cqt, axis=1) + 1e-9))
        print(f"{source_path.name:21s} {text(truth):14s} "
              f"{text(notes(model, cqt, threshold)):14s} "
              f"{text(notes(fft_model, fft, fft_threshold)):14s} {cosine:.3f}")


if __name__ == '__main__':
    main()
