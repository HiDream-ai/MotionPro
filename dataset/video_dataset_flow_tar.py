
import torch
import pytorch_lightning as pl
from vtdm.utils import instantiate_from_config
import webdataset


def webdata_collate_fn(examples):
    sample_num = len(examples)
    C, T, H, W = examples[0]['video'].shape
    video_tensor = torch.stack([example["video"] for example in examples])
    video_tensor = video_tensor.to(memory_format=torch.contiguous_format).float()
    masked_flow_sq = torch.stack([example["flow_ori"] for example in examples])
    masked_flow_sq = masked_flow_sq.to(memory_format=torch.contiguous_format).float()
    
    fps_id = torch.tensor([example["fps_id"] for example in examples])
    fps_id = fps_id.to(memory_format=torch.contiguous_format).float()
    
    motion_bucket_id = torch.stack([example["motion_bucket_id"] for example in examples])
    motion_bucket_id = motion_bucket_id.to(memory_format=torch.contiguous_format).float()
    
    caption = [example["caption"] for example in examples]
    video_name = [example['video_name'] for example in examples]
    
    if "mask" in examples[0]:
        full_mask = torch.stack([example["mask"] for example in examples])
        full_mask = full_mask.to(memory_format=torch.contiguous_format).float()
    else:
        full_mask = None
    
    if "visible_mask_ori" in examples[0]:
        visible_mask_ori = torch.stack([torch.from_numpy(example['visible_mask_ori']) for example in examples])
        visible_mask_ori = visible_mask_ori.to(memory_format=torch.contiguous_format).float()
    else:
        visible_mask_ori = None

    if "flow_brush_mask" in examples[0]:
        flow_brush_mask = torch.stack([example["flow_brush_mask"] for example in examples])
        flow_brush_mask = flow_brush_mask.to(memory_format=torch.contiguous_format).float()
    else:
        flow_brush_mask = None

    if "vis_mask_sq" in examples[0]:
        vis_mask_sq = torch.stack([example["vis_mask_sq"] for example in examples])
        vis_mask_sq = vis_mask_sq.to(memory_format=torch.contiguous_format).float()
    else:
        vis_mask_sq = None

    sample_dict = {
        "video": video_tensor, 'flow_ori': masked_flow_sq, 'mask': full_mask, 'caption': caption, 'fps_id': fps_id, 'motion_bucket_id': motion_bucket_id,
        "video_name": video_name, "flow_brush_mask": flow_brush_mask, 'visible_mask_ori': visible_mask_ori, "vis_mask_sq": vis_mask_sq, 
    }
    
    return sample_dict


class VideoDataset(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, world_size, use_worker_init_fn, dataset_size, train_set, seed=2024, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.use_worker_init_fn = use_worker_init_fn
        self.dataset_size = dataset_size
        self.train_set = train_set
        self.seed = seed
        self.additional_args = kwargs
        
        self.number_of_batches = self.dataset_size[0] // (self.batch_size * self.world_size)

    def setup(self):
        dataset = instantiate_from_config(self.train_set)
        self.train_dataset = (dataset.batched(self.batch_size, partial=False, collation_fn=webdata_collate_fn))

    def train_dataloader(self):
        return webdataset.WebLoader(self.train_dataset, batch_size=None, shuffle=False, num_workers=self.num_workers).with_length(self.number_of_batches)

