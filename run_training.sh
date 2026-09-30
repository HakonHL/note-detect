#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE=${IMAGE:-guitar-ml-training:tf-2.15.1}

cd "$ROOT"
docker build -f Dockerfile.training -t "$IMAGE" .
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$ROOT:/work" \
    "$IMAGE" \
    --dataset dataset_voicings \
    --out "${OUT:-artifacts/chord_cnn}" \
    --test-fraction 0.25 \
    --frames 32 \
    --sample-rate 16000 \
    --epochs 60 \
    --batch-size 32 \
    --seed 42
