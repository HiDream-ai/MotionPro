import argparse, os
import torch
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import shutil as sh
from dataset.video_dataset_flow import WebVid_DOT_28w
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision.utils import flow_to_image
from vtdm.utils import attach_arrows
import datetime
from pathlib import Path
from PIL import Image


def save_sample_results(images, batch, save_dir=None, batch_idx=0, node_rank=0):
    
    batch_masked_flow = images['flow'].permute(0,1,3,4,2)[:,:,:,:,:2]
    batch_mask = images['flow'].permute(0,1,3,4,2)[:,:,:,:,-1]
    b,f,h,w = batch_mask.shape
    
    base_save_dir = os.path.join(save_dir, 'image_log')
    # filename = "results_gs-{:06}_e-{:06}_b-{:06}".format(0, 0, batch_idx)
    filename = 'videos'
    save_video_dir = os.path.join(base_save_dir, filename)
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
                attach_arrows(batch_masked_flow[i,j].detach().cpu().numpy(), base_save_dir, 0, node_rank, idx=j)
                ).to(torch.uint8)
            
            video_frame_list.append(torch.concat([frame1, mask, masked_flow, masked_arrow, video[j].detach().cpu(), predict_video[j].detach().cpu()], dim=1))

        concat_video = torch.stack(video_frame_list)
        video_name = batch['video_name'][i] + '.mp4'
        video_path = os.path.join(save_video_dir, video_name)
        
        torchvision.io.write_video(video_path, concat_video, fps=8, video_codec='h264', options={'crf': '10'})
        
        # save single frames for FID and FVD test..
        pred_img_path = os.path.join(os.path.dirname(save_video_dir), 'single_frames', 'pred')
        gt_img_path = os.path.join(os.path.dirname(save_video_dir), 'single_frames', 'gt')
        os.makedirs(pred_img_path, exist_ok=True)
        os.makedirs(gt_img_path, exist_ok=True)
        
        for f_idx in range(f):
            video_name_id = batch['video_name'][i]
            pred_frame = concat_video[f_idx,:,-w:,:]       # GT and pred videos
            pred_image = Image.fromarray(pred_frame.numpy())
            save_path = os.path.join(pred_img_path, f'{video_name_id}_f_{f_idx:03d}.png')
            pred_image.save(save_path)
            
            gt_frame = concat_video[f_idx,:,-w*2:-w,:]       # GT and pred videos
            gt_image = Image.fromarray(gt_frame.numpy())
            save_path = os.path.join(gt_img_path, f'{video_name_id}_f_{f_idx:03d}.png')
            gt_image.save(save_path)
            


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base", type=str, default='configs/eval/eval_ratio_85.yaml', help="base_model_config")
    parser.add_argument("--output_dir", type=str, default='all_results/eval', help="output dir")
    parser.add_argument("--seed", type=int, default=2025, help="test sampling seed..")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/MotionPro_Sparse-gs_16k.pt', help="ckpt dir")
    
    return parser


def main():

    # --------------------1、read config
    parser = get_parser()
    args = parser.parse_args()
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # output_dir = os.path.join(args.output_dir, f'{now}_test_1')
    config = OmegaConf.load(args.base)
    output_dir = os.path.join(args.output_dir, Path(args.base).stem)
    model_config = config.pop("model", OmegaConf.create())
    dataset_config = config.pop("data", OmegaConf.create()).params
    lighting_configs = config.pop("lightning", OmegaConf.create())
    log_images_kwargs = lighting_configs.callbacks.image_logger.params.log_images_kwargs
    
    node_seed = args.seed
    seed_everything(node_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt_path is not None:
        model_config.params.ckpt_path = args.ckpt_path
        extra_name = args.ckpt_path.split('/')[-2]
        output_dir = os.path.join(output_dir, extra_name)
        
    os.makedirs(output_dir,exist_ok=True)
    sh.copyfile(args.base, os.path.join(output_dir, 'eval_config.yaml'))                # save input config setup
    
    # --------------------2、instant model and dataset
    model = instantiate_from_config(model_config).to(torch.float16)
    model = model.to(device)
    test_dataset = WebVid_DOT_28w(**dataset_config)
    test_dataloader = DataLoader(test_dataset, batch_size=dataset_config['batch_size'], 
                                 shuffle=False, num_workers=dataset_config['num_workers'])
    
    
    model.eval()
    # --------------------3、begain sampling..
    for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        with torch.no_grad():
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device=device, dtype=torch.float16)
            sample_results = model.log_images(batch, split='test', **log_images_kwargs)

        # save the sample_results, NOTE: save as png
        save_sample_results(sample_results, batch, output_dir, batch_idx=idx, node_rank=0)


if __name__ == '__main__':
    main()
