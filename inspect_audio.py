#!/usr/bin/env python3
"""Inspect a rendered chord: play it, show its spectrum, and check the chroma.

The chroma panel is the important one -- it shows whether the pitch classes that
define the chord actually dominate, which is what a chord classifier relies on.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Expected pitch classes per chord, used to sanity-check the chroma.
CHORD_TONES = {
    "C": ["C", "E", "G"],      "Cm": ["C", "D#", "G"],
    "D": ["D", "F#", "A"],     "Dm": ["D", "F", "A"],
    "E": ["E", "G#", "B"],     "Em": ["E", "G", "B"],
    "F": ["F", "A", "C"],      "Fm": ["F", "G#", "C"],
    "G": ["G", "B", "D"],      "Gm": ["G", "A#", "D"],
    "A": ["A", "C#", "E"],     "Am": ["A", "C", "E"],
    "B": ["B", "D#", "F#"],    "Bm": ["B", "D", "F#"],
}


def play(path):
    for player in ("pw-play", "paplay", "aplay"):
        if shutil.which(player):
            subprocess.run([player, str(path)], capture_output=True)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="audio file to inspect")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--save", default=None, help="write the plot to this path")
    parser.add_argument("--label", default=None,
                        help="expected chord name (defaults to the filename prefix)")
    args = parser.parse_args()

    path = Path(args.wav)
    if not path.is_file():
        sys.exit(f"No such file: {path}")

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio, sr = librosa.load(str(path), sr=None, mono=True)
    duration = len(audio) / sr
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))

    print(f"{path.name}: {duration:.2f}s @ {sr} Hz")
    print(f"  peak {peak:.3f}  rms {rms:.4f}  "
          f"({20 * np.log10(rms + 1e-12):.1f} dBFS)")

    # Chroma over the sustained part, skipping the transient attack.
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, fmin=librosa.note_to_hz("E2"))
    start = min(int(0.2 * chroma.shape[1]), chroma.shape[1] - 1)
    mean_chroma = chroma[:, start:].mean(axis=1)
    order = np.argsort(mean_chroma)[::-1]

    label = args.label or path.stem.split("_")[0]
    expected = CHORD_TONES.get(label)

    print("  chroma (strongest first):")
    for i in order[:6]:
        mark = ""
        if expected:
            mark = " <- expected" if PITCH_CLASSES[i] in expected else ""
        print(f"    {PITCH_CLASSES[i]:2s} {mean_chroma[i]:.3f}{mark}")

    if expected:
        top3 = {PITCH_CLASSES[i] for i in order[:3]}
        hits = top3 & set(expected)
        print(f"  expected {label}: {'/'.join(expected)} -> "
              f"{len(hits)}/3 in top 3 {'OK' if len(hits) == 3 else 'CHECK'}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    times = np.arange(len(audio)) / sr
    axes[0].plot(times, audio, linewidth=0.5)
    axes[0].set(title=f"{path.name} - waveform", xlabel="s", ylabel="amplitude")
    axes[0].set_xlim(0, duration)

    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log", ax=axes[1])
    axes[1].set(title="spectrogram (log frequency)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    img = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", ax=axes[2])
    axes[2].set(title="chroma")
    fig.colorbar(img, ax=axes[2])

    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"  plot saved to {args.save}")
    else:
        if not args.no_play and not play(path):
            print("  (no audio player found)")
        plt.show()


if __name__ == "__main__":
    main()
