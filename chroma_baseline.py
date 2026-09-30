#!/usr/bin/env python3
"""Evaluate chord separability with a simple chroma-template classifier.

This is a feasibility baseline, not a trained ML model. It extracts one mean
12-bin CQT chroma vector per WAV, averages training vectors into one template per
chord, then classifies held-out clips by cosine similarity. When voicing metadata
exists, whole voicings are held out so test clips use shapes absent from training.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_labels(dataset: Path):
    csv_path = dataset / "labels.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing label file: {csv_path}; generate the dataset first.")

    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = dataset / row["path"]
            if not path.is_file():
                raise SystemExit(f"Missing audio referenced by labels.csv: {path}")
            rows.append((path, row["label"], row.get("voicing_id", "")))
    if not rows:
        raise SystemExit(f"No entries in {csv_path}")
    return rows


def extract_feature(path: Path, start_s: float, end_s: float | None):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))

    start = min(int(start_s * sr), len(mono))
    end = len(mono) if end_s is None else min(int(end_s * sr), len(mono))
    segment = mono[start:end]
    if segment.size < 2048:
        raise ValueError(f"Analysis segment too short in {path}: {len(segment)} samples")

    # CQT uses semitone-aligned bins. Folding octaves gives the 12 pitch classes.
    chroma = librosa.feature.chroma_cqt(
        y=segment,
        sr=sr,
        fmin=librosa.note_to_hz("C1"),
        n_chroma=12,
        bins_per_octave=36,
        n_octaves=7,
        tuning=0.0,
        norm=2,
    )
    vector = np.mean(chroma, axis=1).astype(np.float64)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector, rms


def stratified_split(rows, test_fraction, seed, holdout_voicings=True):
    by_label = defaultdict(list)
    for item in rows:
        by_label[item[1]].append(item)

    rng = np.random.default_rng(seed)
    train, test = [], []
    for label, items in sorted(by_label.items()):
        if holdout_voicings and label != "none" and all(item[2] for item in items):
            by_voicing = defaultdict(list)
            for item in items:
                by_voicing[item[2]].append(item)
            voicings = sorted(by_voicing)
            if len(voicings) > 1:
                order = rng.permutation(len(voicings))
                n_test = min(len(voicings) - 1,
                             max(1, int(round(len(voicings) * test_fraction))))
                test_voicings = {voicings[i] for i in order[:n_test]}
                for voicing, examples in by_voicing.items():
                    (test if voicing in test_voicings else train).extend(examples)
                continue
        indices = rng.permutation(len(items))
        n_test = max(1, int(round(len(items) * test_fraction))) if len(items) > 1 else 0
        test_ids = set(indices[:n_test])
        for i, item in enumerate(items):
            (test if i in test_ids else train).append(item)
    return train, test


def make_templates(train, features):
    grouped = defaultdict(list)
    for path, label, _voicing in train:
        grouped[label].append(features[path][0])

    templates = {}
    for label, vectors in grouped.items():
        template = np.mean(vectors, axis=0)
        norm = np.linalg.norm(template)
        templates[label] = template / norm if norm else template
    return templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--random-clip-split", action="store_true",
                        help="randomly split clips instead of holding out whole voicings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=float, default=0.15,
                        help="ignore this many seconds at the attack (default: 0.15)")
    parser.add_argument("--end", type=float, default=1.45,
                        help="stop feature window here in seconds (default: 1.45)")
    parser.add_argument("--none-rms", type=float, default=0.01,
                        help="RMS below this is classified as no-chord; set 0 to disable")
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.start < 0 or args.end <= args.start:
        parser.error("require 0 <= --start < --end")

    rows = load_labels(args.dataset)
    labels = sorted({label for _, label, _ in rows})
    print(f"Extracting CQT chroma for {len(rows)} clips across {len(labels)} labels...")

    features = {}
    for i, (path, _label, _voicing) in enumerate(rows, 1):
        try:
            features[path] = extract_feature(path, args.start, args.end)
        except Exception as exc:
            raise SystemExit(f"Feature extraction failed for {path}: {exc}") from exc
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    train, test = stratified_split(rows, args.test_fraction, args.seed,
                                   holdout_voicings=not args.random_clip_split)
    templates = make_templates(train, features)
    chord_labels = sorted(label for label in templates if label != "none")

    # Calibrate a simple energy gate from the training split, if there is a none class.
    none_threshold = args.none_rms
    if "none" in templates and args.none_rms == -1:
        none_rms = [features[path][1] for path, label, _ in train if label == "none"]
        chord_rms = [features[path][1] for path, label, _ in train if label != "none"]
        if none_rms and chord_rms:
            none_threshold = (max(none_rms) + min(chord_rms)) / 2

    predictions = []
    for path, actual, _voicing in test:
        vector, rms = features[path]
        if "none" in templates and none_threshold > 0 and rms < none_threshold:
            predicted = "none"
        else:
            scores = {label: float(np.dot(vector, templates[label])) for label in chord_labels}
            predicted = max(scores, key=scores.get)
        predictions.append((actual, predicted, path))

    total_correct = sum(actual == predicted for actual, predicted, _ in predictions)
    split_name = "random clips" if args.random_clip_split else "held-out voicings where available"
    print(f"\nSplit: {len(train)} train / {len(test)} test ({split_name})")
    print(f"Overall accuracy: {total_correct}/{len(test)} = {total_correct / len(test):.1%}")

    by_label = defaultdict(lambda: [0, 0])
    confusion = Counter()
    for actual, predicted, _path in predictions:
        by_label[actual][1] += 1
        by_label[actual][0] += actual == predicted
        confusion[(actual, predicted)] += 1

    print("\nPer-label accuracy:")
    for label in sorted(by_label):
        correct, count = by_label[label]
        print(f"  {label:5s} {correct:3d}/{count:<3d} {correct / count:6.1%}")

    errors = [(count, actual, predicted) for (actual, predicted), count in confusion.items()
              if actual != predicted]
    if errors:
        print("\nMost common confusions:")
        for count, actual, predicted in sorted(errors, reverse=True)[:15]:
            print(f"  {actual:5s} -> {predicted:5s}: {count}")
    else:
        print("\nNo held-out errors.")

    print("\nNote: the no-chord class uses an RMS gate, not chroma-template matching.")
    if "none" in templates and args.none_rms > 0:
        print(f"RMS gate: {args.none_rms:g} (set --none-rms -1 to calibrate from training data, "
              "or 0 to disable)")
    print("This evaluates synthetic SoundFont clips only; it does not estimate performance "
          "on a physical guitar or pickup.")


if __name__ == "__main__":
    main()
