#!/bin/bash
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES=2
torchrun --standalone --nproc_per_node=1 scripts/generate/image/T2/generate_image_T2.py \
    --net='./training-runs/00000-edm2-img512-control-xs/network-snapshot-0209715-0.100.pkl' \
    --gnet='./training-runs/00000-edm2-img512-control-xs/network-snapshot-0005242-0.050.pkl' \
    --segdir="./synthetic/layout/mask_t2/msk" \
    --pre_latent_dir="./synthetic/image/image_t1" \
    --outdir="./synthetic/image/image_t2" \
    --seeds='0-99' \
    --batch=16 \
