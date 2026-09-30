#!/usr/bin/env python3
"""Train a small Axon-oriented chord CNN and export float/int8 TFLite artifacts.

The input is a time sequence of CQT chroma frames [time, 12, 1]. The neural
network uses only Conv2D, ReLU, MaxPool, Mean/global average, FullyConnected,
and Softmax operators. Data is split by complete voicing groups where metadata
is available, so test voicings are absent from training.

This emulates the quantized network and operator constraints, not the embedded
CQT frontend. That frontend still needs a fixed-point implementation and
feature-parity validation on device.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import librosa
import numpy as np
import soundfile as sf

tf = None



def read_manifest(dataset: Path):
    csv_path = dataset / "labels.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}; generate dataset_voicings first.")
    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = dataset / row["path"]
            if not path.is_file():
                raise SystemExit(f"Missing audio file listed in labels: {path}")
            rows.append({"path": path, "label": row["label"],
                         "voicing_id": row.get("voicing_id", "")})
    if not rows:
        raise SystemExit("Dataset manifest is empty")
    return rows


def split_groups(rows, fraction, seed):
    """Hold out chord voicings; randomize individual no-chord examples."""
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train, test = [], []
    for label, examples in sorted(by_label.items()):
        groups = defaultdict(list)
        for row in examples:
            group = row["voicing_id"]
            if label == "none" or not group:
                group = f"clip:{row['path'].name}"
            groups[group].append(row)

        group_names = sorted(groups)
        if len(group_names) < 2:
            raise SystemExit(f"Need at least two independent groups for class {label!r}")
        order = rng.permutation(len(group_names))
        n_test = min(len(group_names) - 1, max(1, round(len(group_names) * fraction)))
        test_groups = {group_names[i] for i in order[:n_test]}
        for group, group_rows in groups.items():
            (test if group in test_groups else train).extend(group_rows)

    return train, test


def audio_to_feature(path: Path, frames: int, sr_expected: int):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise ValueError(f"{path} is {sr} Hz; expected {sr_expected} Hz")
    return chroma_features(audio.mean(axis=1), sr, frames)


def chroma_features(mono, sr: int, frames: int):
    # CQT chroma frames retain temporal attack/decay and fold octaves into 12 bins.
    hop = sr // 50  # 20 ms frame step at 16 kHz.
    chroma = librosa.feature.chroma_cqt(
        y=mono,
        sr=sr,
        hop_length=hop,
        fmin=librosa.note_to_hz("C1"),
        n_chroma=12,
        bins_per_octave=36,
        n_octaves=7,
        tuning=0.0,
        norm=2,
    ).T

    if len(chroma) < frames:
        chroma = np.pad(chroma, ((0, frames - len(chroma)), (0, 0)))
    else:
        chroma = chroma[:frames]
    return chroma.astype(np.float32)[..., np.newaxis]


def build_model(input_shape, class_count):
    inputs = tf.keras.Input(shape=input_shape, name="chroma_frames")
    x = tf.keras.layers.Conv2D(8, (5, 3), padding="same", use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(12, (3, 3), padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(class_count, name="class_logits")(x)
    outputs = tf.keras.layers.Softmax(name="class_probabilities")(logits)
    model = tf.keras.Model(inputs, outputs, name="guitar_chord_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tflite_predict(model_path, x, integer_input):
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    scale, zero = input_detail["quantization"]
    predictions = []

    for sample in x:
        if integer_input:
            q = np.rint(sample / scale + zero)
            q = np.clip(q, -128, 127).astype(np.int8)
            tensor = q[np.newaxis, ...]
        else:
            tensor = sample[np.newaxis, ...].astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        result = interpreter.get_tensor(output_detail["index"])[0]
        if np.issubdtype(result.dtype, np.integer):
            out_scale, out_zero = output_detail["quantization"]
            result = (result.astype(np.float32) - out_zero) * out_scale
        predictions.append(result)
    return np.asarray(predictions), input_detail, output_detail


def report_metrics(name, y_true, probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == y_true))
    print(f"{name} accuracy: {np.sum(predictions == y_true)}/{len(y_true)} = {accuracy:.1%}")

    counts = Counter(y_true)
    correct = Counter()
    confusion = Counter()
    for actual, predicted in zip(y_true, predictions):
        correct[int(actual)] += int(actual == predicted)
        confusion[(int(actual), int(predicted))] += 1
    print("Per-class:")
    for idx, label in enumerate(labels):
        if counts[idx]:
            print(f"  {label:5s} {correct[idx]:3d}/{counts[idx]:<3d} "
                  f"{correct[idx] / counts[idx]:6.1%}")
    mistakes = [(count, labels[a], labels[p])
                for (a, p), count in confusion.items() if a != p]
    if mistakes:
        print("Top confusions:")
        for count, actual, predicted in sorted(mistakes, reverse=True)[:12]:
            print(f"  {actual:5s} -> {predicted:5s}: {count}")
    return accuracy


def convert_tflite(model, path, representative):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [sample[np.newaxis, ...].astype(np.float32)] for sample in representative
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())


def write_axon_yaml(output_dir: Path, labels):
    import yaml

    y_test = np.load(output_dir / "y_test.npy")
    selected_vectors = []
    seen_classes = set()
    for index, class_id in enumerate(y_test):
        if int(class_id) not in seen_classes:
            seen_classes.add(int(class_id))
            selected_vectors.append(index)

    config = {
        "guitar_chord_cnn": {
            "tflite_model": "chord_cnn_int8.tflite",
            "float_model": "chord_cnn_float.h5",
            "model_name": "GUITAR_CHORD_CNN",
            "train_data": None,
            "test_data": "x_test.npy",
            "test_labels": "y_test.npy",
            "test_labels_format": "just_labels",
            "classification_labels": labels,
            "test_vectors": selected_vectors,
            "header_file_test_vector_cnt": len(selected_vectors),
            "interlayer_buffer_size": 120000,
            "psum_buffer_size": 180000,
            "get_quantized_data": False,
            "run_all_variants": False,
            "conv2d_setting": "local_psum",
            "psum_buffer_placement": "interlayer_buffer",
            "normalize_scaleshift": True,
            "disable_op_quantization": True,
            "skip_softmax_op": False,
            "op_radix": 0,
            "log_level": "info",
        }
    }
    (output_dir / "axon_compile.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_voicings"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/chord_cnn"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.frames < 16 or args.frames % 4:
        parser.error("--frames must be >=16 and divisible by 4 for the pooling stack")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")

    global tf
    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    rows = read_manifest(args.dataset)
    labels = sorted({row["label"] for row in rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    development_rows, test_rows = split_groups(rows, args.test_fraction, args.seed)
    train_rows, validation_rows = split_groups(development_rows, 0.17, args.seed + 1)
    print(f"Group split: {len(train_rows)} train / {len(validation_rows)} validation / "
          f"{len(test_rows)} final held-out clips")
    print(f"Feature window: {args.frames} frames x 20 ms = {args.frames * 20} ms "
          "before inference (not including compute/decision delay)")
    print(f"Classes ({len(labels)}): {', '.join(labels)}")

    cache = {}
    for i, row in enumerate(rows, 1):
        cache[row["path"]] = audio_to_feature(row["path"], args.frames, args.sample_rate)
        if i % 100 == 0 or i == len(rows):
            print(f"  extracted {i}/{len(rows)} feature tensors")

    def arrays(examples):
        x = np.stack([cache[row["path"]] for row in examples]).astype(np.float32)
        y = np.asarray([label_to_id[row["label"]] for row in examples], dtype=np.int32)
        return x, y

    x_train, y_train = arrays(train_rows)
    x_validation, y_validation = arrays(validation_rows)
    x_test, y_test = arrays(test_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "x_test.npy", x_test)
    np.save(args.out / "y_test.npy", y_test)
    np.save(args.out / "class_names.npy", np.asarray(labels))
    (args.out / "split.json").write_text(json.dumps({
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_method": "voicing-grouped; no-chord examples split by clip",
        "train": [{"path": str(row["path"].relative_to(args.dataset)),
                   "label": row["label"], "voicing_id": row["voicing_id"]}
                  for row in train_rows],
        "validation": [{"path": str(row["path"].relative_to(args.dataset)),
                        "label": row["label"], "voicing_id": row["voicing_id"]}
                       for row in validation_rows],
        "test": [{"path": str(row["path"].relative_to(args.dataset)),
                  "label": row["label"], "voicing_id": row["voicing_id"]}
                 for row in test_rows],
    }, indent=2), encoding="utf-8")

    model = build_model(x_train.shape[1:], len(labels))
    model.summary()
    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(x_validation, y_validation),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=2,
    )
    float_loss, float_acc = model.evaluate(x_test, y_test, verbose=0)
    float_probs = model.predict(x_test, verbose=0)
    print(f"\nKeras held-out loss {float_loss:.4f}")
    report_metrics("Keras float", y_test, float_probs, labels)

    float_model_path = args.out / "chord_cnn_float.tflite"
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model_path.write_bytes(float_converter.convert())
    model.save(args.out / "chord_cnn_float.h5", include_optimizer=False)

    int8_model_path = args.out / "chord_cnn_int8.tflite"
    convert_tflite(model, int8_model_path, x_train)
    quantized_probs, input_detail, output_detail = tflite_predict(
        int8_model_path, x_test, integer_input=True)
    report_metrics("TFLite int8", y_test, quantized_probs, labels)
    print("TFLite int8 input:", input_detail["shape"], input_detail["dtype"],
          "scale/zero-point", input_detail["quantization"])
    print("TFLite int8 output:", output_detail["dtype"],
          "scale/zero-point", output_detail["quantization"])
    print(f"\nArtifacts written to {args.out}")
    print(f"  float TFLite: {float_model_path.stat().st_size:,} bytes")
    print(f"  int8 TFLite:  {int8_model_path.stat().st_size:,} bytes")

    write_axon_yaml(args.out, labels)
    print(f"  Axon compiler YAML: {args.out / 'axon_compile.yaml'}")
    print("Next: run the Axon operator scanner/compiler on the int8 TFLite model. "
          "The host CQT frontend is not yet deployed on the MCU.")


if __name__ == "__main__":
    main()
