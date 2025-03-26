import sys
sys.dont_write_bytecode = True      # Avoid produce __pycache__ file..

import argparse, os
import torch
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import shutil as sh
from tqdm import tqdm
import torchvision
from torchvision.utils import flow_to_image
from vtdm.utils import attach_arrows
from PIL import Image
import numpy as np
import glob
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('tools/co-tracker')
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path



def convert_pred_points_into_flow(pred_tracks, video_interp, pred_visibility):

    opt_flow = (pred_tracks - pred_tracks[0][None])
    opt_flow = opt_flow.permute(1,0,2)
    vis_mask_sq = pred_visibility.unsqueeze(-1).permute(1,0,2)
    n,t,c = opt_flow.shape
    _,_,h,w = video_interp.shape

    opt_final = torch.zeros((t,h,w,2)).permute(1,2,0,3)
    vis_mask_final = torch.zeros((t,h,w,1), dtype=torch.bool).permute(1,2,0,3)
    for i in range(n):
        coord = (int(pred_tracks[0, i, 0]), int(pred_tracks[0, i, 1]))
        opt_final[coord[1], coord[0]] = opt_flow[i]
        vis_mask_final[coord[1], coord[0]] = vis_mask_sq[i]

    return opt_final.permute(2,0,1,3), vis_mask_final.permute(2,0,1,3)


def save_sample_results(images, batch, save_video_dir=None, batch_idx=0, node_rank=0, save_name=None):
    
    batch_masked_flow = images['flow'].permute(0,1,3,4,2)[:,:,:,:,:2]
    batch_mask = images['flow'].permute(0,1,3,4,2)[:,:,:,:,-1]
    b,f,h,w = batch_mask.shape
    
    os.makedirs(save_video_dir, exist_ok=True)
    
    for i in range(b):
        video_frame_list = []
        video = torch.clamp(images['reconstructions-video'].permute(0,2,3,4,1)[i], -1., 1.)/2 + 0.5
        video = (video * 255).to(torch.uint8)           # [f,h,w,c], torch.uint8
        predict_video = torch.clamp(images['samples-video'].permute(0,2,3,4,1)[i], -1., 1.)/2 + 0.5
        predict_video = (predict_video * 255).to(torch.uint8)           # [f,h,w,c], torch.uint8
        
        for j in range(f):
            frame1 = video[0].detach().cpu()
            masked_flow = flow_to_image(batch_masked_flow[i,j].to(torch.float32).unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze().detach().cpu()
            mask = (batch_mask[i,j]*255).to(torch.uint8).unsqueeze(-1).repeat(1,1,3).detach().cpu()
            
            masked_arrow = torch.from_numpy(
                attach_arrows(batch_masked_flow[i,j].detach().cpu().numpy(), save_video_dir, 0, node_rank, idx=j)
                ).to(torch.uint8)
            
            video_frame_list.append(torch.concat([frame1, mask, masked_flow, masked_arrow, video[j].detach().cpu(), predict_video[j].detach().cpu()], dim=1))

        concat_video = torch.stack(video_frame_list)
        save_name = save_name.replace('.mp4', '')
        video_name = batch['video_name'] + f'_{save_name}.mp4'
        video_path = os.path.join(save_video_dir, video_name)
        torchvision.io.write_video(video_path, concat_video, fps=8, video_codec='h264', options={'crf': '10'})
        
        video_name = batch['video_name'] + f'_{save_name}-single.mp4'
        video_path = os.path.join(save_video_dir, video_name)
        torchvision.io.write_video(video_path, concat_video[:, :, -w:, :], fps=8, video_codec='h264', options={'crf': '10'})

            
                

def read_img_as_video(img_path):

    img = torch.from_numpy(np.array(Image.open(img_path)))
    img = (img / 255) * 2 - 1
    video = img.unsqueeze(0).repeat(16,1,1,1)

    return video

def read_mask_sq(mask_sq_dir):
    h,w = 320, 512
    mask_file_path_list = sorted(glob.glob(mask_sq_dir + '/*.jpg'))
    visible_np_list = []
    for i in range(len(mask_file_path_list)):
        if i==0:
            mask_numpy = np.ones((h,w), dtype=np.float32)
            mask_numpy_final = np.ones((h,w), dtype=np.float32)
        else:
            ori_mask_numpy = np.array(Image.open(mask_file_path_list[i-1])).astype(np.float32)
            mask_numpy = np.zeros_like(ori_mask_numpy, dtype=np.float32)
            mask_numpy[ori_mask_numpy>125.0] = 1.0 
            
            mask_numpy_final = np.logical_and(mask_numpy_final, mask_numpy).astype(np.float32)
        
        visible_np_list.append(mask_numpy)
    
    vis_mask_sq = torch.from_numpy(np.stack(visible_np_list)).unsqueeze(-1)

    return vis_mask_sq


def read_both_visible_mask(camera_flow_path=None, obj_flow_path=None):
    if camera_flow_path is not None:
        camera_mask_path = os.path.join(os.path.dirname(camera_flow_path), 'visible_mask')
        mask_sq_cam = read_mask_sq(camera_mask_path)
    
    if obj_flow_path is not None:
        obj_mask_path = os.path.join(os.path.dirname(obj_flow_path), 'visible_mask')
        mask_sq_obj = read_mask_sq(obj_mask_path)

    if camera_flow_path is not None and obj_flow_path is not None:
        mask_insert = (mask_sq_cam.int() & mask_sq_obj.int()).float()
    elif camera_flow_path is not None:
        mask_insert = mask_sq_cam
    else:
        mask_insert = mask_sq_obj

    return mask_insert

def interpolate_video(video, ori_f, new_f, size):
    
    video_interp = F.interpolate(video[0], size, mode="bilinear")[None]
    indices = torch.linspace(0, ori_f - 1, new_f).long()
    video_interp_new = video_interp[:, indices, :, :, :].repeat(1,1,1,1,1)

    return video_interp_new


def save_cotracker_results(pred_tracks, pred_visibility, output_path, video_interp):
    b,f,c,h,w = video_interp.shape
    opt_flow, vis_mask_final = convert_pred_points_into_flow(pred_tracks[0], video_interp[0], pred_visibility[0])
    opt_flow_np = opt_flow.squeeze().detach().cpu().numpy().astype(np.float16)

    flo_file_path = os.path.join(output_path, 'optical_flow.npy')
    np.save(flo_file_path, opt_flow_np)


    visible_masks = vis_mask_final.reshape(f,h,w,1).unsqueeze(0)

    for j in range(f):
        mask_numpy = visible_masks[0,j].squeeze().detach().cpu().numpy().astype(np.uint8)*255
        visible_mask_path = os.path.join(output_path, 'visible_mask')
        os.makedirs(visible_mask_path, exist_ok=True)
        mask_file_path = os.path.join(visible_mask_path, f'visible_mask_{j+1:02d}.jpg')
        img = Image.fromarray(mask_numpy)
        img.save(mask_file_path)


    video_unint = video_interp.permute(0,1,3,4,2).squeeze().to(torch.uint8).detach().cpu()
    target_path = os.path.join(output_path, 'video.mp4')
    torchvision.io.write_video(
        target_path, 
        video_unint, 
        fps=8, video_codec='h264', options={'crf': '10'}
    )

    first_frame = Image.fromarray(video_unint[0].detach().cpu().numpy())
    first_frame.save(os.path.join(output_path, 'first_frame.png'))


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='configs/inference_all_flow_from_svd.yaml', help="config file")
    parser.add_argument("--seed", type=int, default=2025, help="test sampling seed..")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/MotionPro_Dense-gs_14k.pt', help="ckpt path")
    parser.add_argument("--ori_video", type=str, default='assets/cases/dog_pure_obj_motion.mp4', help="ori video path")
    parser.add_argument("--camera_video", type=str, default='assets/cases/dog_pure_camera_motion_2.mp4', help="pure camera video path")
    parser.add_argument("--save_name", type=str, default='syn_video.mp4', help="final video save name")
    parser.add_argument("--run_cotracker", type=bool, default=True, help="run cotracker")
    
    return parser


def main():
    # --------- 1. Read config
    parser = get_parser()
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    lighting_configs = config.pop("lightning", OmegaConf.create())
    log_images_kwargs = lighting_configs.callbacks.image_logger.params.log_images_kwargs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    seed_everything(args.seed)

    base_results_dir = 'all_results/test'
    # --------- 2. Stage_1: Get motion flow from Cotracker
    if args.run_cotracker:
        cotrack = CoTrackerPredictor(
        checkpoint='./tools/co-tracker/checkpoints/scaled_offline.pth'
        )
        cotrack = cotrack.to(device)
        cotrack.eval()
    
        for i, video_path in enumerate([args.ori_video, args.camera_video]):
            assert video_path is not None, 'video path is None'
            output_path = os.path.join(base_results_dir , f"{args.save_name.replace('.mp4', '')}",'cotracker', os.path.basename(video_path).replace('.mp4', ''))
            os.makedirs(output_path, exist_ok=True)
            
            video = read_video_from_path(video_path)
            video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)
            _,ori_f,_,_,_ = video.shape
        
            video_interp_new = interpolate_video(video, ori_f, 16, [320, 512])
            b,f,c,h,w = video_interp_new.shape

            grid_query_frame=0
            pred_tracks, pred_visibility = cotrack(video_interp_new, grid_query_frame=grid_query_frame, backward_tracking=False)
            
            save_cotracker_results(pred_tracks, pred_visibility, output_path, video_interp_new)
            
            if i==0:
                obj_flow_path=os.path.join(output_path, 'optical_flow.npy')
                first_frame_path=os.path.join(output_path, 'first_frame.png')
            else:
                camera_flow_path=os.path.join(output_path, 'optical_flow.npy')
    else:
        for i, video_path in enumerate([args.ori_video, args.camera_video]):
            assert video_path is not None, 'video path is None'
            output_path = os.path.join(base_results_dir, f"{args.save_name.replace('.mp4', '')}",'cotracker', os.path.basename(video_path).replace('.mp4', ''))
            
            if i==0:
                obj_flow_path=os.path.join(output_path, 'optical_flow.npy')
                first_frame_path=os.path.join(output_path, 'first_frame.png')
                assert os.path.exists(obj_flow_path) and os.path.exists(first_frame_path), 'obj flow path or first frame path does not exist'
            else:
                camera_flow_path=os.path.join(output_path, 'optical_flow.npy')
                assert os.path.exists(camera_flow_path), 'camera flow path does not exist'
    
    # --------- 3. Stage_2: Run MotionPro-Dense model according to the combined motion flow
    if args.ckpt_path is not None:
        model_config['params']['ckpt_path'] = args.ckpt_path
    
    model = instantiate_from_config(model_config).to(torch.float16)
    model = model.to(device)
    model.eval()

    base_save_dir = os.path.join(base_results_dir, f"{args.save_name.replace('.mp4', '')}")
    motion_type = 'combined'
    video_name = 'final'

    # read all flows
    camera_flow = torch.from_numpy(np.load(camera_flow_path).astype(np.float32))
    obj_flow = torch.from_numpy(np.load(obj_flow_path).astype(np.float32))

    # read visible mask
    if motion_type=='only_obj':
        final_flow = obj_flow + camera_flow*0
        vis_mask_sq = read_both_visible_mask(obj_flow_path=obj_flow_path)
    elif motion_type=='only_cam':
        final_flow = obj_flow*0 + camera_flow
        vis_mask_sq = read_both_visible_mask(camera_flow_path=camera_flow_path)
    else:
        final_flow = obj_flow + camera_flow
        vis_mask_sq = read_both_visible_mask(camera_flow_path, obj_flow_path)
    
    
    motion_bucket_id = torch.mean(torch.sum(final_flow**2, dim=-1).sqrt())      # ! NOTE
    frames = read_img_as_video(first_frame_path)


    batch = {}
    batch['video'] = frames.unsqueeze(0).to(dtype=dtype, device=device).permute(0,4,1,2,3)
    batch['flow_ori'] = final_flow.unsqueeze(0).to(dtype=dtype, device=device)
    batch['vis_mask_sq'] = vis_mask_sq.unsqueeze(0).to(dtype=dtype, device=device)
    batch['caption'] = None
    batch['fps_id'] = torch.tensor(8).unsqueeze(0).to(dtype=dtype, device=device)
    batch['motion_bucket_id'] = motion_bucket_id.unsqueeze(0).to(dtype=dtype, device=device)
    batch["video_name"] = video_name

    # -----------Start sampling
    sample_results = model.log_images(batch, split='test', use_optimization=True, **log_images_kwargs)
    save_sample_results(sample_results, batch, base_save_dir, batch_idx=0, node_rank=0, save_name=args.save_name)

if __name__ == '__main__':
    main()
