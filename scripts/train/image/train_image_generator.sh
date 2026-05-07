export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/train/image/train_image_generator.py \
    --outdir=training-runs/00000-edm2-img512-control-xs \
    --data=./datasets \
    --preset=edm2-img512-xs \
    --batch-gpu=8 \
    --batch=32 \
    --status='10Ki'\
    --snapshot='10Ki'\
    --checkpoint='10Ki'\
    --lr=0.001 \
    --duration='100Ki' \
    --dropout=0.1


