import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
import random
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import glob
import pandas as pd
from decord import VideoReader, cpu
import cv2
import torch.nn.functional as F


class WebVid_DOT_28w(Dataset):
    """
    load processed dataset..
    Assumes webvid data is structured as follows.
    folders/
        - 1066674877
            --flow_01.npy, ....,
            --visible_mask_01.jpg, ...,
            --video.mp4
        - 1066675123
        - 1066675198
        - ...
    """
    def __init__(self,
                 meta_path,
                 resolution=[320, 512],
                 block_size=8,
                 sample_frames=16,
                 fps_id=8,
                 random_mask_ratio=[0.75, 1.00],
                 trigger_word='',
                 **kwargs,
                 ):

        self.meta_path = meta_path
        self.resolution = resolution
        self.block_size = block_size
        self.sample_frames = sample_frames
        self.fps_id = fps_id
        self.random_mask_ratio = random_mask_ratio
        self.trigger_word = trigger_word

        self.video_list = [os.path.join(self.meta_path, video_name) for video_name in sorted(os.listdir(self.meta_path))]
        
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        
        index = index % len(self.video_list)
        video_folder = self.video_list[index]
        video_path, caption = os.path.join(video_folder, 'video.mp4'), ''+self.trigger_word
        video_name = video_path.split('/')[-3] + '-' + video_path.split('/')[-2]
        
        # 1、read video
        video_reader = VideoReader(video_path, ctx=cpu(0))
        frame_indices = list(range(0, len(video_reader)))
        frames = video_reader.get_batch(frame_indices)
        
        frames = torch.tensor(frames.asnumpy()).permute(0,3,1,2).float()     # [t,h,w,c] -> [t,c,h,w]
        t, c, h, w = frames.shape
        
        if self.resolution is not None:
            assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        video = frames / 255.0 * 2.0 - 1.0             # convert into [-1, 1]-------DONE. add change the shape as: [t, c, h, w]
        
        # 2、get optical flow and visible mask area
        flow_np_list = []
        flow_file_path_list = sorted(glob.glob(video_folder + '/*.npy'))
        mask_file_path_list = sorted(glob.glob(video_folder + '/*.jpg'))
        for i in range(t):
            if i==0:
                flow_numpy = np.zeros((h,w,2), dtype=np.float32)
                mask_numpy_final = np.ones((h,w), dtype=np.float32)
            else:
                flow_numpy = np.load(flow_file_path_list[i-1]).astype(np.float32)
                ori_mask_numpy = np.array(Image.open(mask_file_path_list[i-1])).astype(np.float32)
                mask_numpy = np.zeros_like(ori_mask_numpy, dtype=np.float32)
                mask_numpy[ori_mask_numpy>125.0] = 1.0 
                
                mask_numpy_final = np.logical_and(mask_numpy_final, mask_numpy).astype(np.float32)
                
            flow_np_list.append(flow_numpy)
        
        flow_sq = torch.from_numpy(np.stack(flow_np_list))
        
        # 3、get brush area with motion..
        flow_sum_square = torch.zeros(h,w)
        for i in range(t):
            flow_sum_square += torch.sum(flow_sq[i]**2, dim=-1)
            
        flow_brush_mask = flow_sum_square/t > 1.0
        flow_brush_mask = flow_brush_mask[None,:,:,None].repeat(t,1,1,1)
        
        # compute motion_bucket
        motion_bucket_id = torch.mean(torch.sum(flow_sq**2, dim=-1).sqrt())
        # process optical flow and mask into high masked ratio 8*8 block formation 
        count_flag = 0
        
        while True: 
            mask_ratio = random.uniform(min(self.random_mask_ratio), max(self.random_mask_ratio))
            
            block_mask = np.random.rand(h//self.block_size, w//self.block_size) > mask_ratio
            mask_numpy_final_resized = cv2.resize(mask_numpy_final, (w//self.block_size, h//self.block_size), interpolation=cv2.INTER_NEAREST) 
            
            block_mask_new = np.logical_and(block_mask, mask_numpy_final_resized).astype(np.float32)
            full_mask = np.kron(block_mask_new, np.ones((self.block_size, self.block_size), dtype=np.uint8))
            full_mask = torch.from_numpy(full_mask).to(torch.float32)[None,:,:,None].repeat(t,1,1,1)
            
            masked_flow_sq = flow_sq * full_mask
            if torch.sum(full_mask)!=0:
                break
            else:
                count_flag = count_flag+1
            if count_flag>3:
                print(f'this video has a bad trajectory..{video_path}')
                return self.__getitem__(random.randint(0, len(self)-1))
        
        data = {
            'video': video.permute(1,0,2,3), 'flow_ori': masked_flow_sq, 'mask': full_mask, 'caption': caption, 
            'fps_id': self.fps_id, 'motion_bucket_id': motion_bucket_id, "video_name": video_name,
            "flow_brush_mask": flow_brush_mask, 'visible_mask_ori': mask_numpy_final, 'data_type': 'sparse',
        }
        return data
        

class VideoDataset(pl.LightningDataModule):
    def __init__(self, meta_path, resolution, block_size, sample_frames, fps_id,
                 random_mask_ratio, batch_size, num_workers=4, seed=0, **kwargs):
        super().__init__()
        
        self.meta_path = meta_path
        self.resolution = resolution
        self.block_size = block_size
        self.sample_frames = sample_frames
        self.fps_id = fps_id
        self.random_mask_ratio = random_mask_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.additional_args = kwargs

    def setup(self):
        self.train_dataset = WebVid_DOT_28w(self.meta_path, self.resolution, self.block_size, self.sample_frames, self.fps_id,
                                            self.random_mask_ratio)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)




# debug dataset...

# def tensor_to_mp4(video, savepath, fps, rescale=True, nrow=None):
#     """
#     video: torch.Tensor, b,c,t,h,w, 0-1
#     if -1~1, enable rescale=True
#     """
#     n = video.shape[0]
#     video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
#     nrow = int(np.sqrt(n)) if nrow is None else nrow
#     frame_grids = [torchvision.utils.make_grid(framesheet, nrow=nrow) for framesheet in video] # [3, grid_h, grid_w]
#     grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
#     grid = torch.clamp(grid.float(), -1., 1.)
#     if rescale:
#         grid = (grid + 1.0) / 2.0
#     grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
#     #print(f'Save video to {savepath}')
#     torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})


# dataset = WebVid_DOT_28w(meta_path='/mnt/afsdata/zhongwei_dataset/webvid_dot', discretize_optical_flow=True)    # all use default params
# dataloader = torch.utils.data.DataLoader(        
#     dataset,
#     shuffle=False,
#     batch_size=2,
#     num_workers=0,
# )

# save_dir = 'all_results/test_dataset'
# os.makedirs(save_dir, exist_ok=True)

# motion_score = []
# for idx, data in enumerate(dataloader):
#     print('dataset iter....')
    
#     for i in range(2):
#         motion_score.append(data['motion_bucket_id'][i])
        
#     if idx>5000:
#         break
    
# # print(f'ave motion_score is: {sum(motion_score)/len(motion_score)}')     # the results is: 17.63

# # * --------------test_1: for visualization..
#     visible_mask = data['visible_mask']
#     block_mask_ori = data['block_mask_ori']
#     block_mask_new = data['block_mask_new']
#     visible_mask_ori = data['visible_mask_ori']
#     mask = data['mask']

#     all_mask_1 = torch.concat([visible_mask, block_mask_ori, block_mask_new], dim=2)
#     all_mask_2 = torch.concat([visible_mask_ori, mask[:,0].squeeze()], dim=2)
#     video = data['video']
    
#     for i in range(2):
#         tensor_to_mp4(video[i:i+1], savepath=os.path.join(save_dir, f'{idx:02d}_video_{i:02d}.mp4'), fps=8)
#         save_mask = all_mask_1[i].squeeze()
#         img = Image.fromarray(
#             save_mask.detach().cpu().to(torch.uint8).numpy()*255
#         )
#         img.save(os.path.join(save_dir, f'{idx:02d}_frame_mask_1_{i:02d}_ori_visi_mask_and_block_mask_and_final_mask.png'))
        
#         save_mask = all_mask_2[i].squeeze()
#         img = Image.fromarray(
#             save_mask.detach().cpu().to(torch.uint8).numpy()*255
#         )
#         img.save(os.path.join(save_dir, f'{idx:02d}_frame_mask_2_{i:02d}_ori_visi_mask_and_final_mask.png'))
