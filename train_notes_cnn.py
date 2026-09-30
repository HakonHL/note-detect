#!/usr/bin/env python3
"""Train a multi-label pitch-class CNN (12 outputs, C..B) on dataset_notes.

Same Axon-friendly CQT-chroma CNN as train_chord_cnn.py, but the head is 12
independent logits trained with sigmoid cross-entropy. The sigmoid/threshold is
applied outside the model (on the MCU CPU), so the exported model ends at the
Dense layer. Clips are split by complete voicing, stratified by category.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import soundfile as sf

from train_chord_cnn import chroma_features

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
tf = None


def read_manifest(dataset):
    rows = []
    with (dataset / "labels.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            target = np.zeros(12, dtype=np.float32)
            for name in row["pitch_classes"].split():
                target[NOTE_NAMES.index(name)] = 1.0
            rows.append({"path": dataset / row["path"], "rel": row["path"],
                         "category": row["category"], "voicing_id": row["voicing_id"],
                         "label": row["label"], "target": target})
    if not rows:
        raise SystemExit("Dataset manifest is empty")
    return rows


def split_groups(rows, fraction, rng):
    """Hold out whole voicings per category; none clips are grouped individually."""
    by_category = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = row["voicing_id"] if row["category"] != "none" else row["rel"]
        by_category[row["category"]][group].append(row)
    keep, held = [], []
    for category in sorted(by_category):
        groups = sorted(by_category[category])
        order = rng.permutation(len(groups))
        n_held = max(1, round(len(groups) * fraction))
        held_groups = {groups[i] for i in order[:n_held]}
        for group in groups:
            (held if group in held_groups else keep).extend(by_category[category][group])
    return keep, held


def extract(args_tuple):
    path, frames, sr_expected = args_tuple
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise ValueError(f"{path} is {sr} Hz; expected {sr_expected} Hz")
    return chroma_features(audio.mean(axis=1), sr, frames)


def load_features(rows, frames, sr, cache_path, workers):
    key = [row["rel"] for row in rows]
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if (list(cached["paths"]) == key and int(cached["frames"]) == frames
            and ("sample_rate" not in cached or int(cached["sample_rate"]) == sr)):
            print(f"Loaded cached features from {cache_path}")
            return cached["x"]
    jobs = [(str(row["path"]), frames, sr) for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        features = []
        for i, feature in enumerate(pool.map(extract, jobs, chunksize=16), 1):
            features.append(feature)
            if i % 500 == 0 or i == len(jobs):
                print(f"  extracted {i}/{len(jobs)} feature tensors")
    x = np.stack(features).astype(np.float32)
    np.savez(cache_path, x=x, paths=np.asarray(key), frames=frames, sample_rate=sr)
    return x


def build_model(input_shape, width, arch):
    inputs = tf.keras.Input(shape=input_shape, name="chroma_frames")
    x = inputs
    if arch == "gap":
        for i, filters in enumerate((8, 12, 16)):
            kernel = (5, 3) if i == 0 else (3, 3)
            x = tf.keras.layers.Conv2D(filters * width, kernel, padding="same")(x)
            x = tf.keras.layers.ReLU()(x)
            if i < 2:
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        # Pool over time only, so each of the 12 pitch-class columns survives to the head.
        for i, filters in enumerate((8, 12, 16)):
            kernel = (5, 3) if i == 0 else (3, 3)
            x = tf.keras.layers.Conv2D(filters * width, kernel, padding="same")(x)
            x = tf.keras.layers.ReLU()(x)
            if i < 2:
                x = tf.keras.layers.MaxPooling2D((2, 1))(x)
        x = tf.keras.layers.AveragePooling2D((x.shape[1], 1))(x)
        x = tf.keras.layers.Flatten()(x)
    logits = tf.keras.layers.Dense(12, name="pitch_class_logits")(x)
    model = tf.keras.Model(inputs, logits, name="guitar_note_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="bit_acc")],
    )
    return model


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def exact_match(y, probs, threshold):
    return float(np.mean(np.all((probs >= threshold) == (y > 0.5), axis=1)))


def best_threshold(y, probs):
    candidates = np.round(np.arange(0.2, 0.81, 0.05), 2)
    return max(candidates, key=lambda t: exact_match(y, probs, t))


def report(name, y, probs, threshold, categories):
    pred = probs >= threshold
    truth = y > 0.5
    tp = np.sum(pred & truth, axis=0)
    fp = np.sum(pred & ~truth, axis=0)
    fn = np.sum(~pred & truth, axis=0)
    micro_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)
    exact = np.all(pred == truth, axis=1)
    print(f"\n{name} (threshold {threshold:.2f}): exact set match {exact.mean():.1%}, "
          f"note precision {micro_p:.1%}, recall {micro_r:.1%}, F1 {micro_f1:.3f}")
    print("  per note  " + " ".join(f"{n:>5s}" for n in NOTE_NAMES))
    print("  precision " + " ".join(f"{tp[i] / max(tp[i] + fp[i], 1):5.0%}" for i in range(12)))
    print("  recall    " + " ".join(f"{tp[i] / max(tp[i] + fn[i], 1):5.0%}" for i in range(12)))
    print("  by category: exact match, extra notes/clip, missed notes/clip")
    for category in sorted(set(categories)):
        mask = np.asarray(categories) == category
        extra = np.sum(pred[mask] & ~truth[mask]) / mask.sum()
        missed = np.sum(~pred[mask] & truth[mask]) / mask.sum()
        print(f"    {category:8s} n={mask.sum():4d}  {exact[mask].mean():6.1%}  "
              f"+{extra:.2f}  -{missed:.2f}")
    return float(exact.mean())


def tflite_logits(model_path, x):
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    logits = []
    for sample in x:
        q = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[np.newaxis])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        logits.append((raw - out_zero) * out_scale)
    return np.asarray(logits), inp, out


def write_axon_yaml(out_dir, test_categories):
    import yaml

    # One reference vector per category; multi-label targets have no classification labels.
    vectors, seen = [], set()
    for index, category in enumerate(test_categories):
        if category not in seen:
            seen.add(category)
            vectors.append(index)
    config = {"guitar_note_cnn": {
        "tflite_model": "note_cnn_int8.tflite",
        "float_model": "note_cnn_float.h5",
        "model_name": "GUITAR_NOTE_CNN",
        "train_data": None,
        "test_data": "x_test.npy",
        "test_labels": None,
        "test_labels_format": None,
        "classification_labels": None,
        "test_vectors": vectors,
        "header_file_test_vector_cnt": len(vectors),
        "interlayer_buffer_size": 120000,
        "psum_buffer_size": 180000,
        "get_quantized_data": False,
        "run_all_variants": False,
        "conv2d_setting": "local_psum",
        "psum_buffer_placement": "interlayer_buffer",
        "normalize_scaleshift": True,
        "disable_op_quantization": True,
        "op_radix": 0,
        "log_level": "info",
    }}
    (out_dir / "axon_compile.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                               encoding="utf-8")
    return vectors


def converter_for_model(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    @tf.function(input_signature=[tf.TensorSpec([1, 32, 12, 1], tf.float32,
                                                name="chroma_frames")])
    def infer(features):
        return model(features, training=False)

    frozen = convert_variables_to_constants_v2(infer.get_concrete_function())
    return tf.lite.TFLiteConverter.from_concrete_functions([frozen])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_notes"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/note_cnn"))
    parser.add_argument("--feature", choices=("cqt", "fft"), default="cqt")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--arch", choices=("pitch", "gap"), default="pitch",
                        help="pitch: keep the 12 pitch columns to the head; "
                             "gap: original global-average-pool head")
    parser.add_argument("--width", type=int, default=2,
                        help="multiplier on conv filter counts (8/12/16)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.frames < 16 or args.frames % 4:
        parser.error("--frames must be >=16 and divisible by 4 for the pooling stack")

    global tf
    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    rows = read_manifest(args.dataset)
    rng = np.random.default_rng(args.seed)
    development, test = split_groups(rows, args.test_fraction, rng)
    train, validation = split_groups(development, args.validation_fraction, rng)
    print(f"Voicing-grouped split: {len(train)} train / {len(validation)} validation / "
          f"{len(test)} test clips")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.feature == "fft":
        from fft_chroma import chroma_frames

        key = [row["rel"] for row in rows]
        cache_path = args.out / "fft_features_cache.npz"
        if cache_path.is_file():
            cache = np.load(cache_path, allow_pickle=False)
            if list(cache["paths"]) != key:
                raise SystemExit("FFT feature cache does not match the dataset manifest")
            x_all = cache["x"]
        else:
            features = []
            for i, row in enumerate(rows, 1):
                audio, sr = sf.read(row["path"], dtype="float32", always_2d=True)
                if sr != args.sample_rate:
                    raise SystemExit(f"Unexpected sample rate in {row['path']}")
                features.append(chroma_frames(audio.mean(axis=1), args.frames))
                if i % 500 == 0 or i == len(rows):
                    print(f"  extracted {i}/{len(rows)} FFT features")
            x_all = np.stack(features)
            np.savez(cache_path, x=x_all, paths=np.asarray(key))
    else:
        x_all = load_features(rows, args.frames, args.sample_rate,
                              args.out / "features_cache.npz", args.workers)
    index = {row["rel"]: i for i, row in enumerate(rows)}

    def arrays(subset):
        idx = [index[row["rel"]] for row in subset]
        return x_all[idx], np.stack([row["target"] for row in subset])

    x_train, y_train = arrays(train)
    x_val, y_val = arrays(validation)
    x_test, y_test = arrays(test)
    val_categories = [row["category"] for row in validation]
    test_categories = [row["category"] for row in test]

    np.save(args.out / "x_test.npy", x_test)
    np.save(args.out / "y_test.npy", y_test)
    np.save(args.out / "class_names.npy", np.asarray(NOTE_NAMES))
    (args.out / "split.json").write_text(json.dumps({
        "seed": args.seed,
        "split_method": "voicing-grouped, stratified by category; none split by clip",
        **{name: [{"path": row["rel"], "label": row["label"], "category": row["category"],
                   "voicing_id": row["voicing_id"]} for row in subset]
           for name, subset in (("train", train), ("validation", validation), ("test", test))},
    }, indent=2), encoding="utf-8")

    model = build_model(x_train.shape[1:], args.width, args.arch)
    model.summary()
    model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                                                 min_lr=5e-5),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                             restore_best_weights=True),
        ],
        verbose=2,
    )

    threshold = best_threshold(y_val, sigmoid(model.predict(x_val, verbose=0)))
    report("Keras float validation", y_val, sigmoid(model.predict(x_val, verbose=0)),
           threshold, val_categories)
    report("Keras float test", y_test, sigmoid(model.predict(x_test, verbose=0)),
           threshold, test_categories)

    model.save(args.out / "note_cnn_float.keras", include_optimizer=False)
    (args.out / "note_cnn_float.tflite").write_bytes(converter_for_model(model).convert())

    converter = converter_for_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [sample[np.newaxis].astype(np.float32)] for sample in x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_path = args.out / "note_cnn_int8.tflite"
    int8_path.write_bytes(converter.convert())

    logits, inp, out = tflite_logits(int8_path, x_test)
    report("TFLite int8 test", y_test, sigmoid(logits), threshold, test_categories)
    print("\nint8 input:", inp["shape"], "scale/zero-point", inp["quantization"])
    print("int8 output logits: scale/zero-point", out["quantization"])

    (args.out / "postprocess.json").write_text(json.dumps({
        "outputs": NOTE_NAMES,
        "probability_threshold": float(threshold),
        "feature": args.feature,
        # Equivalent test on the raw logit avoids a sigmoid on the MCU.
        "logit_threshold": float(np.log(threshold / (1 - threshold))),
        "output_quantization": [float(v) for v in out["quantization"]],
    }, indent=2), encoding="utf-8")
    print(f"Artifacts written to {args.out} (threshold in postprocess.json)")
    vectors = write_axon_yaml(args.out, test_categories)
    config_path = args.out / "axon_compile.yaml"
    if not (args.out / "note_cnn_float.h5").is_file():
        text = config_path.read_text().replace("float_model: note_cnn_float.h5",
                                               "float_model: null")
        config_path.write_text(text)
    print(f"Axon compiler YAML: {args.out / 'axon_compile.yaml'} (test vectors {vectors})")


if __name__ == "__main__":
    main()
