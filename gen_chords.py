#!/usr/bin/env python3
"""Generate a labelled guitar chord dataset by rendering MIDI through FluidSynth.

Chords use real guitar voicings (fret positions on standard tuning) rather than
abstract root-position triads, so the harmonic content matches what a pickup
would actually see.
"""

import argparse
import csv
import itertools
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

# Standard tuning, low to high, as MIDI note numbers.
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4

# Fret per string, low to high. None means the string is not played.
CHORD_SHAPES = {
    "C":  [None, 3, 2, 0, 1, 0],
    "Cm": [None, 3, 5, 5, 4, 3],
    "D":  [None, None, 0, 2, 3, 2],
    "Dm": [None, None, 0, 2, 3, 1],
    "E":  [0, 2, 2, 1, 0, 0],
    "Em": [0, 2, 2, 0, 0, 0],
    "F":  [1, 3, 3, 2, 1, 1],
    "Fm": [1, 3, 3, 1, 1, 1],
    "G":  [3, 2, 0, 0, 0, 3],
    "Gm": [3, 5, 5, 3, 3, 3],
    "A":  [None, 0, 2, 2, 2, 0],
    "Am": [None, 0, 2, 2, 1, 0],
    "B":  [None, 2, 4, 4, 4, 2],
    "Bm": [None, 2, 4, 4, 3, 2],
}

# General MIDI guitar programs worth using for an electric-guitar target.
GUITAR_PROGRAMS = {
    "clean": 27,       # Electric Guitar (clean)
    "jazz": 26,        # Electric Guitar (jazz)
    "muted": 28,       # Electric Guitar (muted)
    "overdriven": 29,
    "distortion": 30,
    "steel": 25,       # Acoustic Guitar (steel)
    "nylon": 24,       # Acoustic Guitar (nylon)
}

SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
]


def find_soundfont(explicit=None):
    if explicit:
        if not Path(explicit).is_file():
            sys.exit(f"SoundFont not found: {explicit}")
        return explicit
    for path in SOUNDFONT_CANDIDATES:
        if Path(path).is_file():
            return path
    sys.exit(
        "No SoundFont found. Install one with:\n"
        "  sudo apt install fluid-soundfont-gm\n"
        "or pass --sf2 /path/to/font.sf2"
    )


def chord_to_notes(name):
    """Map a chord shape to (midi_note, string_index) pairs."""
    shape = CHORD_SHAPES[name]
    return [
        (OPEN_STRINGS[i] + fret, i)
        for i, fret in enumerate(shape)
        if fret is not None
    ]


def fret_string(frets):
    return ".".join("x" if fret is None else str(fret) for fret in frets)


def fretboard_voicings(name, max_fret=15):
    """Enumerate playable voicings containing exactly the chord's pitch classes."""
    target_pcs = {pitch % 12 for pitch, _ in chord_to_notes(name)}
    found = {}

    for position in range(0, max_fret + 1, 2):
        per_string = []
        for open_note in OPEN_STRINGS:
            options = [None]
            options.extend(
                fret for fret in range(position, min(position + 4, max_fret) + 1)
                if (open_note + fret) % 12 in target_pcs
            )
            per_string.append(options)

        for frets in itertools.product(*per_string):
            played = [(i, fret, OPEN_STRINGS[i] + fret)
                      for i, fret in enumerate(frets) if fret is not None]
            if not 3 <= len(played) <= 6:
                continue
            pitches = [pitch for _, _, pitch in played]
            if {pitch % 12 for pitch in pitches} != target_pcs:
                continue
            fretted = [fret for _, fret, _ in played if fret > 0]
            if fretted and max(fretted) - min(fretted) > 4:
                continue
            found[tuple(frets)] = pitches

    original = tuple(CHORD_SHAPES[name])
    if original not in found:
        found[original] = [OPEN_STRINGS[i] + fret for i, fret in enumerate(original)
                           if fret is not None]

    ordered = sorted(found, key=lambda frets: tuple(-1 if fret is None else fret
                                                    for fret in frets))
    return [{"frets": frets, "notes": tuple(found[frets]),
             "voicing_id": fret_string(frets)} for frets in ordered]


def select_voicings(name, limit):
    """Choose register-, inversion-, and spacing-diverse shapes deterministically."""
    remaining = fretboard_voicings(name)
    selected = []
    root_pc = min(pitch for pitch, _ in chord_to_notes(name)) % 12
    target_pcs = sorted({pitch % 12 for pitch, _ in chord_to_notes(name)})

    # Seed each available inversion, preferring the familiar open shape for root position.
    for bass_pc in target_pcs:
        same_inversion = [item for item in remaining if min(item["notes"]) % 12 == bass_pc]
        if not same_inversion or len(selected) >= limit:
            continue
        original_frets = tuple(CHORD_SHAPES[name])
        if bass_pc == root_pc:
            chosen = next((item for item in same_inversion
                           if item["frets"] == original_frets), None)
        else:
            chosen = None
        if chosen is None:
            chosen = max(same_inversion,
                         key=lambda item: max(item["notes"]) - min(item["notes"]))
        selected.append(chosen)
        remaining.remove(chosen)

    while remaining and len(selected) < limit:
        def diversity(candidate):
            notes = candidate["notes"]
            bass_pc = min(notes) % 12
            octave = min(notes) // 12
            wide = max(notes) - min(notes) >= 19
            score = 4 * (bass_pc != root_pc) + 3 * wide
            if selected:
                score += min(abs(min(notes) - min(item["notes"]))
                             for item in selected) / 12
                score += 2 * (octave not in {min(item["notes"]) // 12
                                             for item in selected})
                score += 2 * (wide not in {max(item["notes"]) - min(item["notes"]) >= 19
                                           for item in selected})
                score += 3 * (bass_pc not in {min(item["notes"]) % 12
                                              for item in selected})
            return score

        best = max(remaining, key=diversity)
        selected.append(best)
        remaining.remove(best)
    return selected


def build_chord_midi(name, program, velocity, strum_ms, note_end, rng,
                     upstroke=False, notes=None):
    """Build a PrettyMIDI object for one strummed chord.

    Notes are released at note_end, which should be early enough that the
    natural decay finishes inside the clip rather than being cut off.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    notes = chord_to_notes(name) if notes is None else [(pitch, i) for i, pitch in enumerate(notes)]
    if upstroke:
        notes = notes[::-1]

    for order, (pitch, _string) in enumerate(notes):
        # A strum staggers string onsets; a little jitter keeps it from being robotic.
        start = order * strum_ms / 1000.0 + rng.uniform(0, 0.004)
        vel = int(np.clip(velocity + rng.randint(-8, 9), 1, 127))
        instrument.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=note_end)
        )

    midi.instruments.append(instrument)
    return midi


def render_midi(midi, sf2, sample_rate, gain, tmpdir, reverb=False, chorus=False):
    """Render a PrettyMIDI object to a mono float array via the fluidsynth CLI.

    Reverb and chorus default to off: a guitar pickup feeds the ADC directly, so
    effects would be a mismatch against the real target signal.
    """
    mid_path = Path(tmpdir) / "chord.mid"
    wav_path = Path(tmpdir) / "chord.wav"
    midi.write(str(mid_path))

    result = subprocess.run(
        [
            "fluidsynth", "-ni", "-q",
            "-F", str(wav_path),
            "-r", str(sample_rate),
            "-g", str(gain),
            "-R", "1" if reverb else "0",
            "-C", "1" if chorus else "0",
            sf2, str(mid_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not wav_path.is_file():
        sys.exit(f"fluidsynth failed:\n{result.stderr}")

    audio, sr = sf.read(str(wav_path), always_2d=True)
    return audio.mean(axis=1), sr


def postprocess(audio, sample_rate, duration, normalize, fade_ms=30.0):
    """Trim or pad to a fixed length, normalise, then fade out.

    FluidSynth renders well past the note-off (release plus reverb), so trimming
    lands mid-ring and would leave a click unless the tail is faded.
    """
    target_len = int(duration * sample_rate)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    audio = audio[:target_len]

    if normalize:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.7  # about -3 dBFS

    fade_len = min(int(fade_ms / 1000.0 * sample_rate), len(audio))
    if fade_len > 1:
        ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
        audio[-fade_len:] *= ramp

    return audio.astype(np.float32)


def play(path):
    for player in ("pw-play", "paplay", "aplay"):
        if shutil.which(player):
            subprocess.run([player, str(path)], capture_output=True)
            return
    print("  (no audio player found; install pipewire-utils or alsa-utils)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dataset", help="output directory")
    parser.add_argument("--count", type=int, default=20,
                        help="audio variations per chord voicing")
    parser.add_argument("--voicings", type=int, default=8,
                        help="distinct playable voicings per chord")
    parser.add_argument("--sr", type=int, default=16000, help="sample rate in Hz")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="clip length in seconds")
    parser.add_argument("--sf2", default=None, help="SoundFont path")
    parser.add_argument("--gain", type=float, default=0.6, help="fluidsynth gain")
    parser.add_argument("--tones", nargs="+", default=["clean"],
                        choices=sorted(GUITAR_PROGRAMS), help="guitar tones to render")
    parser.add_argument("--chords", nargs="+", default=sorted(CHORD_SHAPES),
                        help="chords to generate")
    parser.add_argument("--silence", type=int, default=20,
                        help="number of low-level noise/no-chord clips")
    parser.add_argument("--hard-negatives", type=int, default=40,
                        help="number of loud single-note/dyad no-chord clips")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--release", type=float, default=0.4,
                        help="seconds before the clip end at which notes are released, "
                             "so the decay finishes inside the clip")
    parser.add_argument("--reverb", action="store_true", help="enable fluidsynth reverb")
    parser.add_argument("--chorus", action="store_true", help="enable fluidsynth chorus")
    parser.add_argument("--fade-ms", type=float, default=20.0,
                        help="fade-out length in ms, as insurance against a residual click")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="render and play N random chords instead of a dataset")
    args = parser.parse_args()

    if args.count < 1 or args.voicings < 1 or args.silence < 0 or args.hard_negatives < 0:
        parser.error("--count and --voicings must be positive; negative counts cannot be negative")

    if not shutil.which("fluidsynth"):
        sys.exit("fluidsynth not found. Install with: sudo apt install fluidsynth")

    sf2 = find_soundfont(args.sf2)
    rng = random.Random(args.seed)
    print(f"SoundFont: {sf2}")

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.preview:
            preview(args, sf2, rng, tmpdir)
            return
        build_dataset(args, sf2, rng, tmpdir)


def render_one(name, args, sf2, rng, tmpdir):
    """Render a single randomised variation of one chord."""
    tone = rng.choice(args.tones)
    velocity = rng.randint(70, 115)
    strum_ms = rng.uniform(8, 35)
    upstroke = rng.random() < 0.3
    note_end = max(0.2, args.duration - args.release)
    voicing = rng.choice(select_voicings(name, args.voicings))

    midi = build_chord_midi(
        name, GUITAR_PROGRAMS[tone], velocity, strum_ms, note_end, rng, upstroke,
        voicing["notes"]
    )
    audio, sr = render_midi(midi, sf2, args.sr, args.gain, tmpdir,
                            args.reverb, args.chorus)
    audio = postprocess(audio, sr, args.duration, not args.no_normalize, args.fade_ms)
    return audio, sr, tone, velocity, strum_ms, upstroke, voicing


def preview(args, sf2, rng, tmpdir):
    names = [rng.choice(args.chords) for _ in range(args.preview)]
    out = Path(tmpdir) / "preview.wav"
    for name in names:
        audio, sr, tone, vel, strum, up, voicing = render_one(
            name, args, sf2, rng, tmpdir)
        sf.write(str(out), audio, sr)
        rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
        print(f"  {name:3s}  tone={tone:10s} vel={vel:3d} "
              f"strum={strum:4.1f}ms {'up' if up else 'down':4s} "
              f"voicing={voicing['voicing_id']} peak={np.abs(audio).max():.2f} rms={rms:.3f}")
        play(out)


def build_dataset(args, sf2, rng, tmpdir):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in args.chords:
        chord_dir = out_dir / name
        chord_dir.mkdir(exist_ok=True)
        voicings = select_voicings(name, args.voicings)
        for voicing_index, voicing in enumerate(voicings):
            for i in range(args.count):
                tone = rng.choice(args.tones)
                velocity = rng.randint(70, 115)
                strum = rng.uniform(8, 35)
                upstroke = rng.random() < 0.3
                note_end = max(0.2, args.duration - args.release)
                midi = build_chord_midi(
                    name, GUITAR_PROGRAMS[tone], velocity, strum, note_end, rng,
                    upstroke, voicing["notes"])
                audio, sr = render_midi(midi, sf2, args.sr, args.gain, tmpdir,
                                        args.reverb, args.chorus)
                audio = postprocess(audio, sr, args.duration, not args.no_normalize,
                                    args.fade_ms)
                path = chord_dir / f"{name}_v{voicing_index:02d}_{i:04d}.wav"
                sf.write(str(path), audio, sr)
                rows.append({
                    "path": str(path.relative_to(out_dir)),
                    "label": name,
                    "voicing_id": f"{name}:{voicing['voicing_id']}",
                    "frets": voicing["voicing_id"],
                    "midi_notes": " ".join(map(str, voicing["notes"])),
                    "tone": tone,
                    "velocity": velocity,
                    "strum_ms": round(strum, 2),
                    "upstroke": int(upstroke),
                })
        print(f"  {name:3s}: {len(voicings)} voicings x {args.count} variations")

    if args.silence:
        silence_dir = out_dir / "none"
        silence_dir.mkdir(exist_ok=True)
        n = int(args.duration * args.sr)
        for i in range(args.silence):
            # Low-level noise, so "no chord" is not trivially separable by energy alone.
            audio = (np.random.default_rng(args.seed + i).standard_normal(n)
                     * 10 ** (-50 / 20)).astype(np.float32)
            audio = postprocess(audio, args.sr, args.duration, False, args.fade_ms)
            path = silence_dir / f"none_{i:04d}.wav"
            sf.write(str(path), audio, args.sr)
            rows.append({"path": str(path.relative_to(out_dir)), "label": "none",
                         "voicing_id": "none", "frets": "", "midi_notes": "",
                         "tone": "-", "velocity": 0, "strum_ms": 0, "upstroke": 0})
        print(f"  none: {args.silence} clips")

    if args.hard_negatives:
        hard_dir = out_dir / "none"
        hard_dir.mkdir(exist_ok=True)
        for i in range(args.hard_negatives):
            chord = rng.choice(args.chords)
            source_notes = [pitch for pitch, _ in chord_to_notes(chord)]
            note_count = rng.choice((1, 2))
            partial_notes = rng.sample(source_notes, note_count)
            tone = rng.choice(args.tones)
            velocity = rng.randint(65, 115)
            strum = rng.uniform(8, 35)
            note_end = max(0.2, args.duration - args.release)
            midi = build_chord_midi(chord, GUITAR_PROGRAMS[tone], velocity, strum,
                                    note_end, rng, notes=partial_notes)
            audio, sr = render_midi(midi, sf2, args.sr, args.gain, tmpdir,
                                    args.reverb, args.chorus)
            audio = postprocess(audio, sr, args.duration, not args.no_normalize,
                                args.fade_ms)
            path = hard_dir / f"none_hard_{i:04d}.wav"
            sf.write(str(path), audio, sr)
            rows.append({
                "path": str(path.relative_to(out_dir)),
                "label": "none",
                "voicing_id": f"none:hard:{i:04d}",
                "frets": "",
                "midi_notes": " ".join(map(str, partial_notes)),
                "tone": tone,
                "velocity": velocity,
                "strum_ms": round(strum, 2),
                "upstroke": 0,
            })
        print(f"  hard negatives: {args.hard_negatives} single-note/dyad clips")

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} clips at {args.sr} Hz -> {out_dir}/  (labels in {csv_path.name})")


if __name__ == "__main__":
    main()
