#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
CUDA_VISIBLE_DEVICES=2 
torchrun --nnodes=1 --nproc_per_node=1 ./preprocessing/encoder_latent.py --data-path ./datasets --features-path ./datasets --image-size 512
