# 1.1 Sample video on MC-Bench
python eval_mc_bench.py \
    --ckpt_path checkpoints/MotionPro_Sparse-gs_16k.pt \
    --dataset_path data/MC-Bench \
    --seed 2025 \
    --output_dir all_results/eval/mc_bench

# 1.2 Eval MD-Video
cd tools/co-tracker
pip install -e .
cd ..
bash eval_co_tracker.sh


# 2.1 Sample 1000 videos from webvid to eval fvd
python eval_fvd.py \
    --base configs/eval/eval_ratio_85.yaml \
    --output_dir all_results/eval/fvd \
    --seed 2025 \
    --ckpt_path checkpoints/MotionPro_Sparse-gs_16k.pt


# 2.2 To evaluate FVD, we use the code from https://github.com/Wangt-CN/DisCo to avoid potential issues and uncertainties.


