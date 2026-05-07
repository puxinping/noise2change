#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
OUTDIR="${OUTDIR:-training-runs/00000-edm2-img512-control-xs}"
DATASET="${DATASET:-./datasets}"
PRETRAINED="${PRETRAINED:-./pretrained_model/edm2-img512-xs-2147483-0.135.pkl}"
PRESET="${PRESET:-edm2-img512-xs}"
BATCH_GPU="${BATCH_GPU:-8}"
BATCH="${BATCH:-32}"
STATUS="${STATUS:-10Ki}"
SNAPSHOT="${SNAPSHOT:-1Mi}"
CHECKPOINT="${CHECKPOINT:-1Mi}"
LR="${LR:-0.001}"
DURATION="${DURATION:-100Ki}"
FP16="${FP16:-True}"
DROPOUT="${DROPOUT:-0.1}"

if [ "${NPROC_PER_NODE}" -lt 1 ] || [ "${NPROC_PER_NODE}" -gt 4 ]; then
    echo "NPROC_PER_NODE must be between 1 and 4, got ${NPROC_PER_NODE}" >&2
    exit 1
fi

VISIBLE_GPU_COUNT=$(python -c 'import os; visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip(); print(len([x for x in visible.split(",") if x.strip()]) if visible else 0)')
if [ "${VISIBLE_GPU_COUNT}" -gt 0 ] && [ "${VISIBLE_GPU_COUNT}" -ne "${NPROC_PER_NODE}" ]; then
    echo "CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPU_COUNT} GPU(s), but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 1
fi

LAUNCHER=(python)
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    LAUNCHER=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
fi

"${LAUNCHER[@]}" scripts/train/image/train_image_generator.py \
    --outdir="${OUTDIR}" \
    --data="${DATASET}" \
    --pretrained="${PRETRAINED}" \
    --preset="${PRESET}" \
    --batch-gpu="${BATCH_GPU}" \
    --batch="${BATCH}" \
    --status="${STATUS}" \
    --snapshot="${SNAPSHOT}" \
    --checkpoint="${CHECKPOINT}" \
    --lr="${LR}" \
    --duration="${DURATION}" \
    --fp16="${FP16}" \
    --dropout="${DROPOUT}"

