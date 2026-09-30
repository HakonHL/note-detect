#!/usr/bin/env python3
"""Generate a pitch-class (note-set) dataset of guitar sounds via FluidSynth.

Each clip is a random, playable guitar voicing of a pitch-class set: single
notes (optionally octave-doubled), intervals, power chords, triads including
inversions, 4-note chords, and non-traditional clusters/random sets. Labels
are the octave-folded pitch classes present, for multi-label training.
"""

import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from gen_chords import (GUITAR_PROGRAMS, OPEN_STRINGS, build_chord_midi, find_soundfont,
                        fret_string, play, postprocess, render_midi)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# category -> {name: (intervals above root, min notes, max notes)}
SETS = {
    "single": {"note": ((0,), 1, 1), "octaves": ((0,), 2, 3)},
    "interval": {
        "m2": ((0, 1), 2, 2), "M2": ((0, 2), 2, 2), "m3": ((0, 3), 2, 3),
        "M3": ((0, 4), 2, 3), "P4": ((0, 5), 2, 3), "TT": ((0, 6), 2, 3),
        "m6": ((0, 8), 2, 3), "M6": ((0, 9), 2, 3), "m7": ((0, 10), 2, 3),
        "M7": ((0, 11), 2, 3),
    },
    "power": {"5": ((0, 7), 2, 4)},
    "triad": {
        "maj": ((0, 4, 7), 3, 6), "min": ((0, 3, 7), 3, 6), "dim": ((0, 3, 6), 3, 5),
        "aug": ((0, 4, 8), 3, 5), "sus2": ((0, 2, 7), 3, 6), "sus4": ((0, 5, 7), 3, 6),
    },
    "tetrad": {
        "7": ((0, 4, 7, 10), 4, 6), "maj7": ((0, 4, 7, 11), 4, 6),
        "m7": ((0, 3, 7, 10), 4, 6), "m7b5": ((0, 3, 6, 10), 4, 5),
        "dim7": ((0, 3, 6, 9), 4, 5), "6": ((0, 4, 7, 9), 4, 6), "m6": ((0, 3, 7, 9), 4, 6),
        "add9": ((0, 2, 4, 7), 4, 5), "madd9": ((0, 2, 3, 7), 4, 5),
        "7sus4": ((0, 5, 7, 10), 4, 6), "mmaj7": ((0, 3, 7, 11), 4, 5),
    },
    "cluster": {
        "c012": ((0, 1, 2), 3, 4), "c013": ((0, 1, 3), 3, 4), "c023": ((0, 2, 3), 3, 4),
        "c014": ((0, 1, 4), 3, 4), "c015": ((0, 1, 5), 3, 4), "c016": ((0, 1, 6), 3, 4),
        "c0123": ((0, 1, 2, 3), 4, 5), "c0145": ((0, 1, 4, 5), 4, 5),
    },
}
# Share of clips per category; "random" draws an arbitrary 3- or 4-note set.
DEFAULT_WEIGHTS = {"single": 0.14, "interval": 0.16, "power": 0.07, "triad": 0.24,
                   "tetrad": 0.16, "cluster": 0.08, "random": 0.10, "none": 0.05}


def sample_voicing(pcs, min_notes, max_notes, rng, max_fret=15, low_strings=False,
                   tries=3000):
    """Randomly search for a playable voicing that contains exactly the pitch classes."""
    pcs = set(pcs)
    for _ in range(tries):
        n = rng.randint(max(min_notes, len(pcs)), min(max_notes, 6))
        # Mostly adjacent strings, sometimes with one muted string inside the block.
        width = n + 1 if 2 < n + 1 <= 6 and rng.random() < 0.2 else n
        lowest = rng.randint(0, min(6 - width, 2) if low_strings else 6 - width)
        strings = list(range(lowest, lowest + width))
        if width > n:
            strings.remove(rng.choice(strings[1:-1]))
        position = rng.randint(0, max_fret - 3)

        frets = [None] * 6
        for string in strings:
            options = [fret for fret in range(position, position + 4)
                       if (OPEN_STRINGS[string] + fret) % 12 in pcs]
            if position > 0 and OPEN_STRINGS[string] % 12 in pcs:
                options.append(0)
            if not options:
                break
            frets[string] = rng.choice(options)
        else:
            notes = [OPEN_STRINGS[s] + frets[s] for s in strings]
            if {note % 12 for note in notes} == pcs and len(set(notes)) == len(notes):
                return frets, notes
    return None


def pick_set(category, rng):
    """Return (label, pitch classes, min notes, max notes, low-strings flag)."""
    root = rng.randrange(12)
    if category == "random":
        size = rng.choice((3, 4))
        pcs = tuple(sorted(rng.sample(range(12), size)))
        return f"rand:{'.'.join(NOTE_NAMES[p] for p in pcs)}", pcs, size, size + 1, False
    name, (intervals, lo, hi) = rng.choice(sorted(SETS[category].items()))
    pcs = tuple(sorted({(root + i) % 12 for i in intervals}))
    return f"{NOTE_NAMES[root]}:{name}", pcs, lo, hi, category == "power"


def plan_clips(args, rng):
    categories = list(DEFAULT_WEIGHTS)
    weights = [DEFAULT_WEIGHTS[c] for c in categories]
    specs = []
    while len(specs) < args.count:
        category = rng.choices(categories, weights)[0]
        spec = {"category": category, "tone": rng.choice(args.tones),
                "velocity": rng.randint(60, 118), "upstroke": rng.random() < 0.3,
                "seed": rng.randrange(2 ** 31)}
        if category == "none":
            spec.update(label="none", pcs=(), frets=None, notes=[], strum_ms=0.0)
            specs.append(spec)
            continue
        label, pcs, lo, hi, low = pick_set(category, rng)
        voicing = sample_voicing(pcs, lo, hi, rng, low_strings=low)
        if voicing is None:
            continue
        frets, notes = voicing
        spec.update(label=label, pcs=pcs, frets=frets, notes=notes,
                    strum_ms=0.0 if len(notes) == 1 else rng.uniform(5, 35))
        specs.append(spec)
    return specs


def render_spec(spec, args, sf2):
    rng = random.Random(spec["seed"])
    if spec["category"] == "none":
        # Low-level broadband noise; real chain idle noise comes from add_idle_noise.py.
        level_db = rng.uniform(-70, -45)
        audio = (np.random.default_rng(spec["seed"]).standard_normal(int(args.duration * args.sr))
                 * 10 ** (level_db / 20)).astype(np.float32)
        return postprocess(audio, args.sr, args.duration, False, args.fade_ms)

    note_end = max(0.2, args.duration - args.release)
    midi = build_chord_midi("", GUITAR_PROGRAMS[spec["tone"]], spec["velocity"],
                            spec["strum_ms"], note_end, rng, spec["upstroke"], spec["notes"])
    with tempfile.TemporaryDirectory() as tmpdir:
        audio, sr = render_midi(midi, sf2, args.sr, args.gain, tmpdir)
    return postprocess(audio, sr, args.duration, True, args.fade_ms)


def row_for(spec, path, out_dir):
    return {
        "path": str(path.relative_to(out_dir)),
        "label": spec["label"],
        "category": spec["category"],
        "voicing_id": f"{spec['label']}:{fret_string(spec['frets'])}" if spec["frets"] else "none",
        "frets": fret_string(spec["frets"]) if spec["frets"] else "",
        "midi_notes": " ".join(map(str, spec["notes"])),
        "pitch_classes": " ".join(NOTE_NAMES[p] for p in spec["pcs"]),
        "tone": spec["tone"] if spec["notes"] else "-",
        "velocity": spec["velocity"] if spec["notes"] else 0,
        "strum_ms": round(spec["strum_ms"], 2),
        "upstroke": int(spec["upstroke"]) if spec["notes"] else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dataset_notes")
    parser.add_argument("--count", type=int, default=4000, help="total clips")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--release", type=float, default=0.4,
                        help="seconds before clip end at which notes are released")
    parser.add_argument("--fade-ms", type=float, default=20.0)
    parser.add_argument("--gain", type=float, default=0.6)
    parser.add_argument("--tones", nargs="+", default=["clean", "jazz", "overdriven", "distortion"],
                        choices=sorted(GUITAR_PROGRAMS))
    parser.add_argument("--sf2", default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="render and play N random clips instead of writing a dataset")
    args = parser.parse_args()

    if not shutil.which("fluidsynth"):
        sys.exit("fluidsynth not found. Install with: sudo apt install fluidsynth")
    sf2 = find_soundfont(args.sf2)
    rng = random.Random(args.seed)

    if args.preview:
        args.count = args.preview
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "preview.wav"
            for spec in plan_clips(args, rng):
                sf.write(out, render_spec(spec, args, sf2), args.sr)
                print(f"  {spec['category']:8s} {spec['label']:16s} "
                      f"notes={spec['notes']} frets={fret_string(spec['frets']) if spec['frets'] else '-'} "
                      f"tone={spec['tone']}")
                play(out)
        return

    specs = plan_clips(args, rng)
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty; remove it or choose another --out")
    paths = []
    counters = Counter()
    for spec in specs:
        category_dir = args.out / spec["category"]
        category_dir.mkdir(parents=True, exist_ok=True)
        paths.append(category_dir / f"{spec['category']}_{counters[spec['category']]:05d}.wav")
        counters[spec["category"]] += 1

    def job(item):
        spec, path = item
        sf.write(path, render_spec(spec, args, sf2), args.sr)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, _ in enumerate(pool.map(job, zip(specs, paths)), 1):
            if done % 250 == 0 or done == len(specs):
                print(f"  rendered {done}/{len(specs)}")

    rows = [row_for(spec, path, args.out) for spec, path in zip(specs, paths)]
    with (args.out / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} clips -> {args.out}")
    for category, count in sorted(counters.items()):
        print(f"  {category:8s} {count}")
    pc_counts = Counter(p for spec in specs for p in spec["pcs"])
    print("Pitch-class occurrences: " +
          " ".join(f"{NOTE_NAMES[p]}={pc_counts[p]}" for p in range(12)))


if __name__ == "__main__":
    main()
