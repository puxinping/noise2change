export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train/image/train_image_generator.py \
    --outdir=training-runs/00000-edm2-img512-control-xs \
    --data=./datasets \
    --preset=edm2-img512-xs \
    --batch-gpu=64 \
    --batch=128 \
    --status='10Ki'\
    --snapshot='1Mi'\
    --checkpoint='1Mi'\
    --lr=0.001 \
    --duration='200Mi' \
    --dropout=0.1


