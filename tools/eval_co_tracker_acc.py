import argparse
import os
from einops import rearrange
from PIL import Image
import torch
import torch.nn.functional as F
import lpips
import clip
from decord import VideoReader, cpu
import yaml
from omegaconf import OmegaConf
from torchvision.transforms import PILToTensor
import torch.nn as nn
import json
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from tqdm import tqdm
import random
import ast
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor


def read_numpy_list_with_stride(video_path, idx_list, width):
    
    video_reader = VideoReader(video_path, ctx=cpu(0))
    frames = video_reader.get_batch(idx_list).asnumpy()
    video_frames_list = [frames[i,:,-width:,:] for i in range(frames.shape[0])]
    
    return video_frames_list

def save_dict_to_yaml(dict_data, file_path):

    conf = OmegaConf.create(dict_data)

    OmegaConf.save(conf, file_path)
    print(f"Dictionary has been saved to {file_path}")

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def interpolate_traj(resized_all_points, model_length=16, select_idx=[0,7,15]):
    
    all_new_points = []
    for splited_track in resized_all_points:
        sub_new_points = []
        if len(splited_track) == 1: # stationary point
            displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
            splited_track = tuple([splited_track[0], displacement_point])
        # interpolate the track
        splited_track = interpolate_trajectory(splited_track, model_length)
        splited_track = splited_track[:model_length]
        if len(splited_track) < model_length:
            splited_track = splited_track + [splited_track[-1]] * (model_length -len(splited_track))
        
        for i in range(len(select_idx)-1):
            sub_new_points.append([splited_track[0], splited_track[select_idx[i+1]]])

        all_new_points.append(sub_new_points)
        
    return all_new_points

def convert_traj_pairs(all_traj_points):
    
    converted_traj_points = []
    for i in range(len(all_traj_points[0])):
        sub_traj = []
        for j in range(len(all_traj_points)):
            sub_traj.append(all_traj_points[j][i])
        
        converted_traj_points.append(sub_traj)
    
    return converted_traj_points

def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

def set_seed(seed=2024):
    random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_path", type=str, help="video_path")         # motion_flow long video
    parser.add_argument("--save_path", type=str, default=None, help="save_path")
    parser.add_argument("--seed", type=int, default=2024, help="seed")
    parser.add_argument("--eval_frame_idx", type=str, default=None, help="which frames to evaluate")
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
    parser.add_argument("--mc_bench_path", type=str, default=None, help="mc_bench_path")
    
    return parser

def get_block_coordinates(x, y, queries_list=None, block_size=8):
    half_block = block_size // 2

    # 计算8x8块的起始和结束坐标
    x_start = max(x - half_block, 0)
    x_end = x + half_block
    y_start = max(y - half_block, 0)
    y_end = y + half_block

    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            queries_list.append([0, i, j])   # [t,x_coord,y_coord]
            # print(f"({i}, {j}) is the {i * block_size + j + 1}-th position from the original ({x}, {y})")
    
    return queries_list

def get_queries_from_traj(traj_points):
    
    queries_list = []
    for splited_track in traj_points: # ref: https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks/demo.ipynb#scrollTo=c6422e7c-8c6f-4269-92c3-245344afe35b
        queries_list = get_block_coordinates(splited_track[0][0][0].astype(int), splited_track[0][0][1].astype(int), queries_list)    # [t,x_coord,y_coord], [h,w]方向上的坐标 ↓ is x_coord, →is y_coord
    
    final_queries = torch.tensor(queries_list).float()
    
    return final_queries


if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mc_bench_path = args.mc_bench_path
    set_seed(args.seed)
    eval_frame_idx = ast.literal_eval(args.eval_frame_idx)
    assert isinstance(eval_frame_idx, list), 'MUST LIST TYPE...'
    
    model = CoTrackerPredictor(checkpoint=args.checkpoint).to(device)
    
    all_video_motion_strength = []
    all_video_dist = []
    all_dict = {}
    final_metrics = {}

    class_name = os.path.basename(args.video_path)
    all_sub_folders = sorted(os.listdir(args.video_path))
    for sub_folder in tqdm(all_sub_folders):
        
        # sub_folder = 'SY_2023-09-15-1741-15_04'     # FOR DEBUG
        if not os.path.isdir(os.path.join(args.video_path, sub_folder)):
            continue

        img_path = os.path.join(args.video_path, sub_folder, 'first_frame.png')
        w, _ = Image.open(img_path).size               # (w,h)
        video_path = os.path.join(args.video_path, sub_folder, args.video_name)
        video = read_video_from_path(video_path)
        video = torch.from_numpy(video[:,:,-w:,:]).permute(0, 3, 1, 2)[None].float().to(device)
        # 1、Get all interpolated Trajectory points pairs...
        config_path = os.path.join(mc_bench_path, class_name, sub_folder, 'meta_data_new.json')
        try:
            with open(config_path, 'r') as f:
                meta_data = json.load(f)
        except:
            continue
        
        resized_all_points = meta_data['traj_key_points']
        all_traj_points = interpolate_traj(resized_all_points, model_length=16, select_idx=eval_frame_idx)   # the same start points is the same group
        
        # 2、Config Tracking point
        # ref: https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks/demo.ipynb#scrollTo=c6422e7c-8c6f-4269-92c3-245344afe35b
        queries = get_queries_from_traj(all_traj_points).to(device)
        pred_tracks, pred_visibility = model(video, queries=queries[None])
        
        # 3、save predict_tracks and visualization video..
        save_dir = os.path.join(args.save_path, sub_folder)
        os.makedirs(save_dir, exist_ok=True)
        vis = Visualizer(
            save_dir=save_dir,
            linewidth=6,
            mode='cool',
            tracks_leave_trace=-1
        )
        vis.visualize(
            video=video,
            tracks=pred_tracks[:,:,36::64,:],
            visibility=pred_visibility,
            filename='queries')
        
        # 文件路径
        file_path = os.path.join(save_dir, 'predict_track_64.pt')
        torch.save(pred_tracks, file_path)
        
        # pred_target_point = torch.mean(pred_tracks, dim=-2)
        pred_target_point = pred_tracks[:,:,36::64,:].detach().cpu()
        
        # 4、Compute drag distance for each drag...
        video_dist = []
        for drag_idx,drag in enumerate(all_traj_points):
        
            for f_idx in range(len(drag)):
                
                GT_target_point = torch.Tensor(drag[f_idx][1])   
                dist = (GT_target_point - pred_target_point[0,f_idx+1,drag_idx]).float().norm()
                video_dist.append(dist)
            
        all_video_dist.append(sum(video_dist) / len(video_dist))
        all_dict[f'{class_name}_{sub_folder}_Co-tracker_DM'] = "{:.5f}".format(sum(video_dist)/len(video_dist))
    
    final_metrics['Total Co-tracker Distance Metrics Results'] = "{:.5f}".format(sum(all_video_dist) / len(all_video_dist))
    final_metrics.update(all_dict)
    
    save_dict_to_yaml(final_metrics, os.path.join(args.save_path, f'Co-tracker_DM_metrics_{len(eval_frame_idx)-1}_pairs.yaml'))
    