#!/usr/bin/env python3
"""Add idle-input captures from the PCM1808 chain to the dataset's "none" class.

Slices a capture (made with i2s_to_mp3.py --capture-only) into clips matching the
dataset format and appends them to labels.csv. Previously added idle clips are
replaced, so the script can be rerun; rerun it after regenerating the dataset.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
PREFIX = "none_idle_"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*",
                        default=[ROOT / "captures" / "pcm1808-idle.raw24.wav"],
                        help="unscaled idle captures (*.raw24.wav)")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_voicings")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--skip-seconds", type=float, default=0.5,
                        help="discard the start of each capture while the chain settles")
    parser.add_argument("--max-clips", type=int, default=30,
                        help="cap to keep the none class from dominating the dataset")
    args = parser.parse_args()

    labels_path = args.dataset / "labels.csv"
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if not Path(row["path"]).name.startswith(PREFIX)]
    out_dir = args.dataset / "none"
    for stale in out_dir.glob(f"{PREFIX}*.wav"):
        stale.unlink()

    clip_len = int(args.clip_seconds * args.sample_rate)
    written = 0
    for capture in args.captures:
        audio, rate = sf.read(capture, dtype="float64", always_2d=True)
        mono = audio[int(args.skip_seconds * rate):].mean(axis=1)
        gcd = np.gcd(rate, args.sample_rate)
        mono = resample_poly(mono, args.sample_rate // gcd, rate // gcd)
        for start in range(0, len(mono) - clip_len + 1, clip_len):
            if written >= args.max_clips:
                break
            path = out_dir / f"{PREFIX}{written:04d}.wav"
            # 24-bit keeps the ~-80 dBFS noise floor from collapsing to a few 16-bit LSBs.
            sf.write(path, mono[start:start + clip_len], args.sample_rate, subtype="PCM_24")
            row = dict.fromkeys(fieldnames, "")
            row.update({"path": str(path.relative_to(args.dataset)), "label": "none",
                        "voicing_id": "none", "tone": "pcm1808-idle",
                        "velocity": "0", "strum_ms": "0", "upstroke": "0"})
            if "category" in row:
                row["category"] = "none"
            rows.append(row)
            written += 1

    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {written} idle clips to {out_dir}; labels.csv now has {len(rows)} rows")


if __name__ == "__main__":
    main()
