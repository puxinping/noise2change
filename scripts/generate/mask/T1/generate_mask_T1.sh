export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES=2
python ./scripts/generate/mask/T1/generate_mask_T1.py \
    --net='./training-runs/00000-edm2-mask512-pixels-xs/network-snapshot-0052428-0.100.pkl' \
    --gnet='./training-runs/00000-edm2-mask512-pixels-xs/network-snapshot-0005242-0.050.pkl' \
    --outdir='./synthetic/layout/mask_t1' \
    --seeds='0-99' \
    --batch=8 \
    --class=$CLS

