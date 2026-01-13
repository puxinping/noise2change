export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train/mask/train_mask_generator.py \
    --outdir=training-runs/00000-edm2-mask512-pixels-xs \
    --data=./datasets \
    --preset=edm2-img512-xs \
    --batch-gpu=128 \
    --batch=256 \
    --status='10Ki'\
    --snapshot='1Mi'\
    --checkpoint='1Mi'\
    --lr=0.001 \
    --duration='50Mi' 
