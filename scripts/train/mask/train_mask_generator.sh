#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUTDIR="${OUTDIR:-training-runs/00000-edm2-mask512-pixels-xs}"
DATASET="${DATASET:-./datasets}"
PRETRAINED="${PRETRAINED:-./pretrained_model/edm2-img512-xs-2147483-0.135.pkl}"
PRESET="${PRESET:-edm2-img512-xs}"
BATCH_GPU="${BATCH_GPU:-2}"
BATCH="${BATCH:-4}"
STATUS="${STATUS:-10Ki}"
SNAPSHOT="${SNAPSHOT:-1Mi}"
CHECKPOINT="${CHECKPOINT:-1Mi}"
LR="${LR:-0.001}"
DURATION="${DURATION:-50Mi}"
FP16="${FP16:-True}"

if [ "${NPROC_PER_NODE}" -eq 1 ]; then
    python scripts/train/mask/train_mask_generator.py \
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
        --fp16="${FP16}"
else
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train/mask/train_mask_generator.py \
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
        --fp16="${FP16}"
fi
