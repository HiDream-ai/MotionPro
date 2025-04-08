script_dir=$(dirname "$0")
echo $script_dir

seed=2025
eval_frame_idx="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"
video_name=sampled_video-motion_bucket_id_17-id_frame.mp4
checkpoint=co-tracker/checkpoints/scaled_offline.pth
mc_bench_path="$script_dir/../data/MC-Bench"


video_path="$script_dir/../all_results/eval/mc_bench/checkpoints/Fine_grained_control"  # or object_control
echo $video_path
CUDA_VISIBLE_DEVICES=0 python eval_co_tracker_acc.py --video_path $video_path --save_path $video_path --mc_bench_path $mc_bench_path --seed $seed --eval_frame_idx $eval_frame_idx --video_name $video_name --checkpoint $checkpoint

video_path="$script_dir/../all_results/eval/mc_bench/checkpoints/object_control"  # or object_control
CUDA_VISIBLE_DEVICES=0 python eval_co_tracker_acc.py --video_path $video_path --save_path $video_path --mc_bench_path $mc_bench_path --seed $seed --eval_frame_idx $eval_frame_idx --video_name $video_name --checkpoint $checkpoint

