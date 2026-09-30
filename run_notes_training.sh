#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE=${IMAGE:-guitar-ml-training:tf-2.15.1}
OUT=${OUT:-artifacts/note_cnn}

cd "$ROOT"
mkdir -p "$OUT"
# Under sudo, hand the output tree back to the invoking user so the container can write it.
if [ -n "${HOST_UID:-}" ]; then
    chown "$HOST_UID:${HOST_GID:-$HOST_UID}" artifacts "$OUT"
fi
docker build -f Dockerfile.training -t "$IMAGE" .
docker run --rm \
    --user "${HOST_UID:-$(id -u)}:${HOST_GID:-$(id -g)}" \
    -v "$ROOT:/work" \
    -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -e HOME=/tmp \
    --entrypoint python \
    "$IMAGE" train_notes_cnn.py \
    --dataset dataset_notes \
    --out "$OUT" \
    "$@"
