export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES=2
python ./scripts/generate/mask/T2/generate_mask_T2.py \
    --net='./training-runs/00000-edm2-mask512-pixels-xs/network-snapshot-0052428-0.100.pkl' \
    --gnet='./training-runs/00000-edm2-mask512-pixels-xs/network-snapshot-0005242-0.050.pkl' \
    --segdir="./synthetic/image/image_t1/msk" \
    --outdir='./synthetic/layout/mask_t2' \
    --seeds='0-99' \
    --batch=8 \
    --class=$CLS
