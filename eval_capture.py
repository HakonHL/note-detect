#!/usr/bin/env python3
"""Evaluate the int8 chord model on audio captured through the PCM1808 chain.

Each played clip listed in the capture's .segments.csv is located in the
recording by cross-correlation with its source WAV. Both the source and the
aligned capture go through the training feature extractor and the int8
TFLite model, so any accuracy difference is caused by the analog/ADC chain.
The silent pre-roll is also scored as a "none" example.
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter
from scipy.signal import correlate, resample_poly

from train_chord_cnn import chroma_features

ROOT = Path(__file__).resolve().parent
# librosa's low CQT octaves warn about n_fft > signal length; training sees the same.
warnings.filterwarnings("ignore", message="n_fft=.*is too large")


class Model:
    def __init__(self, path):
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.frames = int(self.input["shape"][1])

    def __call__(self, feature):
        scale, zero = self.input["quantization"]
        q = np.clip(np.rint(feature / scale + zero), -128, 127).astype(np.int8)
        self.interpreter.set_tensor(self.input["index"], q[np.newaxis])
        self.interpreter.invoke()
        out_scale, out_zero = self.output["quantization"]
        raw = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        return (raw - out_zero) * out_scale


def load_mono(path, sr):
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != sr:
        gcd = np.gcd(rate, sr)
        mono = resample_poly(mono, sr // gcd, rate // gcd)
    return mono


def align(capture, source, nominal_start, sr, search_before, search_after):
    """Return (sample offset of source in capture, normalized correlation peak)."""
    lo = max(0, int((nominal_start - search_before) * sr))
    hi = min(len(capture), int((nominal_start + search_after) * sr) + len(source))
    region = capture[lo:hi]
    template = source[:min(len(source), sr)]
    if len(region) < len(template):
        return None, 0.0
    xcorr = correlate(region, template, mode="valid", method="fft")
    # Normalize by the local energy of the capture so the peak is a correlation coefficient.
    energy = np.sqrt(np.convolve(region ** 2, np.ones(len(template)), mode="valid"))
    ncc = np.abs(xcorr) / (energy * np.linalg.norm(template) + 1e-12)
    best = int(np.argmax(ncc))
    return lo + best, float(ncc[best])


def level_dbfs(audio):
    return 20 * np.log10(np.sqrt(np.mean(np.square(audio))) + 1e-12)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def dataset_entry(path):
    """Return (dataset root, labels.csv row) for a clip, searching its parent folders."""
    for root in path.parents:
        manifest = root / "labels.csv"
        if manifest.is_file():
            rel = str(path.relative_to(root))
            with manifest.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["path"] == rel:
                        return root, row
            return root, None
    return None, None


def note_set(row):
    if row is None:
        return None
    pcs = {int(n) % 12 for n in row.get("midi_notes", "").split()}
    return " ".join(NOTE_NAMES[p] for p in sorted(pcs)) or "none"


def split_membership(split_path):
    if not split_path.is_file():
        return {}
    split = json.loads(split_path.read_text())
    return {entry["path"]: name for name in ("train", "validation", "test")
            for entry in split.get(name, [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="?",
                        default=ROOT / "captures" / "pcm1808-chords.raw24.wav",
                        help="captured WAV (unscaled or normalized)")
    parser.add_argument("--segments", type=Path, default=None,
                        help="segments CSV (default: derived from the capture name)")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "chord_cnn")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--search-before", type=float, default=0.3,
                        help="seconds before the logged start to search for the clip")
    parser.add_argument("--search-after", type=float, default=1.0,
                        help="seconds after the logged start to search (host playback latency)")
    parser.add_argument("--gate-dbfs", type=float, default=-65.0,
                        help="windows quieter than this RMS level are forced to 'none'; "
                             "only meaningful for unscaled (*.raw24.wav) captures")
    args = parser.parse_args()

    segments_path = args.segments
    if segments_path is None:
        stem = args.capture.name.split(".")[0]
        segments_path = args.capture.with_name(f"{stem}.segments.csv")
    with segments_path.open(newline="") as handle:
        segments = list(csv.DictReader(handle))
    if not segments:
        raise SystemExit(f"No segments in {segments_path}")

    sr = args.sample_rate
    postprocess_path = args.model_dir / "postprocess.json"
    notes_mode = postprocess_path.is_file()
    model = Model(next(args.model_dir.glob("*_int8.tflite")))
    if notes_mode:
        threshold = json.loads(postprocess_path.read_text())["probability_threshold"]

        def predict(feature):
            probs = 1.0 / (1.0 + np.exp(-model(feature)))
            names = [NOTE_NAMES[i] for i in range(12) if probs[i] >= threshold]
            # Certainty of the least certain of the 12 on/off decisions.
            confidence = float(np.min(np.maximum(probs, 1 - probs)))
            return " ".join(names) or "none", confidence
    else:
        labels = [str(name) for name in np.load(args.model_dir / "class_names.npy")]

        def predict(feature):
            probs = model(feature)
            return labels[probs.argmax()], float(probs.max())
    membership = split_membership(args.model_dir / "split.json")
    capture = load_mono(args.capture, sr)
    print(f"Capture: {args.capture} ({len(capture) / sr:.2f}s, resampled to {sr} Hz)")
    print(f"Model: {args.model_dir} ({'pitch-class notes' if notes_mode else 'chord classes'}), "
          f"window {model.frames} frames ({model.frames * 20} ms)\n")

    width = 14 if notes_mode else 5
    header = (f"{'clip':22s} {'split':10s} {'truth':{width}s} {'lag ms':>7s} {'xcorr':>6s} "
              f"{'feat cos':>8s} {'dBFS':>6s}  {'source pred':{width + 9}s} capture pred (gated)")
    print(header)
    print("-" * len(header))

    window = int(model.frames * 0.02 * sr)
    source_hits = capture_hits = scored = 0
    first_start = None
    for segment in segments:
        path = Path(segment["file"])
        root, row = dataset_entry(path)
        label = note_set(row) if notes_mode else path.parent.name
        nominal = float(segment["start_s"])
        first_start = nominal if first_start is None else min(first_start, nominal)
        if label is None:
            print(f"{path.name:22s} no labels.csv entry; skipped")
            continue
        source = load_mono(path, sr)
        offset, peak = align(capture, source, nominal, sr, args.search_before, args.search_after)
        if offset is None or offset + len(source) > len(capture):
            print(f"{path.name:22s} not fully captured")
            continue

        source_feature = chroma_features(source.astype(np.float32), sr, model.frames)
        captured_feature = chroma_features(
            capture[offset:offset + len(source)].astype(np.float32), sr, model.frames)
        a, b = source_feature.ravel(), captured_feature.ravel()
        feature_cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        source_pred, source_conf = predict(source_feature)
        capture_pred, capture_conf = predict(captured_feature)
        level = level_dbfs(capture[offset:offset + window])
        if level < args.gate_dbfs:
            capture_pred = "none"
        scored += 1
        source_hits += source_pred == label
        capture_hits += capture_pred == label

        split = membership.get(str(path.relative_to(root)) if root else "", "unknown")
        lag_ms = (offset / sr - nominal) * 1000
        mark = lambda pred: "ok" if pred == label else "XX"
        print(f"{path.name:22s} {split:10s} {label:{width}s} {lag_ms:7.0f} {peak:6.2f} "
              f"{feature_cos:8.3f} {level:6.1f}  "
              f"{source_pred:{width}s} {source_conf:4.0%} {mark(source_pred):3s} "
              f"{capture_pred:{width}s} {capture_conf:4.0%} {mark(capture_pred)}")

    if scored:
        kind = "Exact note-set" if notes_mode else "Chord"
        print(f"\n{kind} accuracy: source {source_hits}/{scored}, capture {capture_hits}/{scored}")

    # The pre-roll before the first clip contains only the chain's idle noise.
    noise_end = int(first_start * sr) if first_start is not None else len(capture)
    starts = range(sr // 10, noise_end - window + 1, sr // 20)
    if not starts:
        print("Pre-roll too short to score idle noise; use a longer --pre-roll when capturing.")
        return
    model_false = gated_false = 0
    levels = []
    for start in starts:
        segment = capture[start:start + window]
        levels.append(level_dbfs(segment))
        pred, _ = predict(chroma_features(segment.astype(np.float32), sr, model.frames))
        model_false += pred != "none"
        gated_false += pred != "none" and levels[-1] >= args.gate_dbfs
    print(f"Idle pre-roll ({len(starts)} windows, {min(levels):.1f}..{max(levels):.1f} dBFS): "
          f"false chords model-only {model_false}/{len(starts)}, "
          f"with {args.gate_dbfs:.0f} dBFS gate {gated_false}/{len(starts)}")


if __name__ == "__main__":
    main()
