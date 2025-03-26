from typing import Any, Dict, List, Union
import torch
from vtdm.utils import instantiate_from_config
import io
import gc
import torch
from typing import Any, Callable, Dict, List, Optional, Union
import webdataset as wds
from webdataset import WebDataset
from .clip_sampling import ClipSampler
from .encoded_video import select_video_class
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random
import cv2
# from .utils_oss import url_opener_oss
import os
import glob


def obtain_tar_list(single_data_path, remove_folder_list):
    check_file_list = os.listdir(single_data_path)
    if len(check_file_list) > 0 and check_file_list[0].endswith('.tar'):
        tar_file_list = glob.glob(os.path.join(single_data_path, '*.tar'))
    else:
        if remove_folder_list is not None:
            tar_file_list = []
            folder_list = os.listdir(single_data_path)
            for folder in folder_list: 
                if folder not in remove_folder_list:
                    file_list = glob.glob(os.path.join(single_data_path, '{}/*.tar'.format(folder)))
                    tar_file_list += file_list
            print('Remove the tar folder list: {}'.format(remove_folder_list))
        else: 
            tar_file_list = glob.glob(os.path.join(single_data_path, '*/*.tar'))    
    return tar_file_list

def obtain_oss_tar_list(data_path=None):
    # tar_name = pd.read_csv(tar_csv_file)['tar_name'].values
    with open(data_path, 'r') as file:
        tar_file_list = [line.strip('\n') for line in file.readlines()]
    
    print(f'the total tar list number is: {len(tar_file_list)}, the total video number is: {len(tar_file_list) * 50}')
    
    return tar_file_list

def get_block_mask_ref_opf_strength(flow_magnitude, mask_ratio=0.2):
    
    # Flatten flow magnitude to 1D array and sort
    h, w = flow_magnitude.shape
    flat_magnitude = flow_magnitude.flatten()
    probabilities = flat_magnitude / flat_magnitude.sum()

    # Get indices of top 20% points
    num_points = int(mask_ratio * flat_magnitude.size)
    sampled_indices = np.random.choice(flat_magnitude.size, size=num_points, p=probabilities, replace=False)
    
    # Convert 1D indices to 2D indices
    sampled_coords = np.unravel_index(sampled_indices, (h, w))

    # Create a mask of zeros
    mask = np.zeros((h, w), dtype=bool)

    # Set positions of top 20% points to 1
    mask[sampled_coords] = True
    
    return mask
    

def create_video_tar_dataset(
    data_path: Union[str, List[str]],
    clip_sampler: ClipSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    decoder: str = "pyav",
    cache_path: str = None,
    resampled: bool = False,
    tokenized: bool = False,
    data_key: List[str] = ['mp4'],
    pretrained_model_name_or_path: str = None,
    select_data_path: str = None,
    remove_folder_list: List[str] = None,
    is_select_csv_file: bool = False,
    random_mask_ratio=None,
    use_oss: bool = False,
    sparse_ref_opf_strength: bool = False,          # add ref_opf_strength

) -> WebDataset:
    
    _MAX_CONSECUTIVE_FAILURES = 10
    if not use_oss:
        assert os.path.isdir(data_path), 'the data path is not a directory: {}'.format(data_path)
        tar_file_list = [os.path.join(data_path, file) for file in sorted(os.listdir(data_path))]
    else:
        assert os.path.isfile(data_path), 'the data path is not a file: {}'.format(data_path)
        tar_file_list = obtain_oss_tar_list(data_path)
        # wds.tariterators.url_opener = url_opener_oss('./oss_cache/aoss_zhangzhongwei.conf')
    
    print('The number of video tar file: {}, the first tar file: {}'.format(len(tar_file_list), tar_file_list[0]))
    video_decoder = select_video_class(decoder)
    web_videodataset = wds.WebDataset(tar_file_list, 
                                      shardshuffle=True,
                                      handler=wds.handlers.warn_and_continue,
                                      resampled=resampled,
                                      nodesplitter=wds.split_by_node).shuffle(1000)
    
    
    def decode_mp4_video_data(item):
        text = item['txt']
        caption = text.decode("utf-8")
        
        # 1、read video
        video_name = item["__key__"]
        if len(data_key) == 1:
            video_data = item[data_key[0]]
        else:
            try:
                for key in list(item.keys()):
                    if key in data_key: 
                        video_data = item[key]
                        break
            except:
                raise ValueError('Can find the key in key list: {}'.format(data_key))
        
        for i_try in range(_MAX_CONSECUTIVE_FAILURES):
            video_bytes = io.BytesIO(video_data)
            video = video_decoder(file=video_bytes, video_name=video_name, decode_video=True, decode_audio=False,)
            (clip_start, clip_end, clip_index, aug_index, is_last_clip,) = clip_sampler(None, video.duration)
            # sampling
            if aug_index == 0:
                _loaded_clip = video.get_clip(clip_start, clip_end)
            # check sampling results
            video_is_null = (_loaded_clip is None or _loaded_clip["video"] is None)
            if (is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                clip_sampler.reset()
                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()
                if video_is_null:
                    print("Failed to load clip {} trial {}".format(video.name, i_try))
            
            frames = _loaded_clip["video"]
            audio_samples = _loaded_clip["audio"]
            
            # 2、get optical flow and visible mask area
            c,t,h,w = frames.shape
            np_mask = np.array(Image.open(io.BytesIO(item['jpg']))).astype(np.float32)
            np_flow = np.load(io.BytesIO(item['npy'])).astype(np.float32)
            flow_np_list = []
            for i in range(t):
                if i==0:
                    flow_numpy = np.zeros((h,w,2), dtype=np.float32)
                    mask_numpy_final = np.ones((h,w), dtype=np.float32)
                else:
                    flow_numpy = np_flow[i-1]
                    ori_mask_numpy = np_mask[:,(i-1)*w:i*w]
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
            block_size = 8
            
            while True: 
                mask_ratio = random.uniform(min(random_mask_ratio), max(random_mask_ratio))
                if sparse_ref_opf_strength:
                    downsampled_flow_sum_square = F.avg_pool2d(torch.sqrt(flow_sum_square).unsqueeze(0).unsqueeze(0), kernel_size=8)
                    flow_magnitude = downsampled_flow_sum_square.squeeze().numpy()
                    
                    block_mask = get_block_mask_ref_opf_strength(flow_magnitude, mask_ratio=1-mask_ratio)
                else:
                    block_mask = np.random.rand(h//block_size, w//block_size) > mask_ratio
                mask_numpy_final_resized = cv2.resize(mask_numpy_final, (w//block_size, h//block_size), interpolation=cv2.INTER_NEAREST) 
            
                block_mask_new = np.logical_and(block_mask, mask_numpy_final_resized).astype(np.float32)
                full_mask = np.kron(block_mask_new, np.ones((block_size, block_size), dtype=np.uint8))
                full_mask = torch.from_numpy(full_mask).to(torch.float32)[None,:,:,None].repeat(t,1,1,1)
                masked_flow_sq = flow_sq * full_mask
                if torch.sum(full_mask)!=0:
                    break
                else:
                    count_flag = count_flag+1
                if count_flag>3:
                    raise RuntimeError(f'this video has a bad trajectory..{video.name}')

            # 4、return all dataset values
            sample_dict = {
                "video": frames, 'flow_ori': masked_flow_sq, 'mask': full_mask, 'caption': caption, 'fps_id': 8, 'motion_bucket_id': motion_bucket_id,
                "video_name": video.name, "flow_brush_mask": flow_brush_mask, 
                **({"audio": audio_samples} if audio_samples is not None else {}),
                'visible_mask_ori': mask_numpy_final
            }
            # only do transformation for videos
            if transform is not None:
                sample_dict = transform(sample_dict)
            
            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {_MAX_CONSECUTIVE_FAILURES} retries."
            )

    transformed_videos_dataset = web_videodataset.map(decode_mp4_video_data, handler=wds.handlers.warn_and_continue)

    return transformed_videos_dataset


def VidTar(
    data_path: Union[str, List[str]],
    clip_sampler: Dict[str, Any],
    transform: Dict[str, Any] = None,
    decode_audio: bool = False,
    decoder: str = "pyav",
    cache_path: str = None,
    resampled: bool = False,
    tokenized: bool = False,
    data_key: List[str] = ['mp4'],
    pretrained_model_name_or_path: str = None,
    select_data_path: str = None, 
    remove_folder_list: List[str] = None,
    is_select_csv_file: bool = False,
    random_mask_ratio=None,
    use_oss: bool = False,
    sparse_ref_opf_strength: bool = False,
):
    """
    A helper function to create ``LabeledVideoDataset`` object for the WebVid dataset.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a directory, the directory structure has data_path/0-9/xxxx-xxxx.tar.

        clip_sampler (Dict[str, Any]): The config to define how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        transform (Dict[str, Any]): Config for the callable function that is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        decoder (str): Defines what type of decoder used to decode a video.

    """

    torch._C._log_api_usage_once("dataset.VidTar")

    clip_sampler = instantiate_from_config(clip_sampler)
    transform = instantiate_from_config(transform)
    
    
    return create_video_tar_dataset(
        data_path,
        clip_sampler,
        transform,
        decoder,
        cache_path,
        resampled,
        tokenized,
        data_key,
        pretrained_model_name_or_path,
        select_data_path,
        remove_folder_list,
        is_select_csv_file,
        random_mask_ratio=random_mask_ratio,
        use_oss=use_oss,
        sparse_ref_opf_strength=sparse_ref_opf_strength,
    )


def create_video_tar_dataset_all_flow(
    data_path: Union[str, List[str]],
    clip_sampler: ClipSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    decoder: str = "pyav",
    resampled: bool = False,
    data_key: List[str] = ['mp4'],
    remove_folder_list: List[str] = None,
    random_mask_ratio=None,
    use_oss: bool = False,

) -> WebDataset:
    
    _MAX_CONSECUTIVE_FAILURES = 10
    if not use_oss:
        assert os.path.isdir(data_path), 'the data path is not a directory: {}'.format(data_path)
        tar_file_list = [os.path.join(data_path, file) for file in sorted(os.listdir(data_path))]
    else:
        assert os.path.isfile(data_path), 'the data path is not a file: {}'.format(data_path)
        tar_file_list = obtain_oss_tar_list(data_path)
        # wds.tariterators.url_opener = url_opener_oss('./oss_cache/aoss_zhangzhongwei.conf')
    
    print('The number of video tar file: {}, the first tar file: {}'.format(len(tar_file_list), tar_file_list[0]))
    video_decoder = select_video_class(decoder)
    web_videodataset = wds.WebDataset(tar_file_list, 
                                      shardshuffle=True,
                                      handler=wds.handlers.warn_and_continue,
                                      resampled=resampled,
                                      nodesplitter=wds.split_by_node).shuffle(1000)


    def decode_mp4_video_data(item):
        text = item['txt']
        caption = text.decode("utf-8")
        
        # 1、read video
        video_name = item["__key__"]
        if len(data_key) == 1:
            video_data = item[data_key[0]]
        else:
            try:
                for key in list(item.keys()):
                    if key in data_key: 
                        video_data = item[key]
                        break
            except:
                raise ValueError('Can find the key in key list: {}'.format(data_key))
        
        for i_try in range(_MAX_CONSECUTIVE_FAILURES):
            video_bytes = io.BytesIO(video_data)
            video = video_decoder(file=video_bytes, video_name=video_name, decode_video=True, decode_audio=False,)
            (clip_start, clip_end, clip_index, aug_index, is_last_clip,) = clip_sampler(None, video.duration)
            # sampling
            if aug_index == 0:
                _loaded_clip = video.get_clip(clip_start, clip_end)
            # check sampling results
            video_is_null = (_loaded_clip is None or _loaded_clip["video"] is None)
            if (is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                clip_sampler.reset()
                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()
                if video_is_null:
                    print("Failed to load clip {} trial {}".format(video.name, i_try))
            
            frames = _loaded_clip["video"]

            # 2、get optical flow and visible mask area
            c,t,h,w = frames.shape
            np_mask = np.array(Image.open(io.BytesIO(item['jpg']))).astype(np.float32)
            np_flow = np.load(io.BytesIO(item['npy'])).astype(np.float32)
            flow_np_list = []
            visible_np_list = []
            for i in range(t):
                if i==0:
                    flow_numpy = np.zeros((h,w,2), dtype=np.float32)
                    mask_numpy = np.ones((h,w), dtype=np.float32)
                    mask_numpy_final = np.ones((h,w), dtype=np.float32)
                else:
                    flow_numpy = np_flow[i-1]
                    ori_mask_numpy = np_mask[:,(i-1)*w:i*w]
                    mask_numpy = np.zeros_like(ori_mask_numpy, dtype=np.float32)
                    mask_numpy[ori_mask_numpy>125.0] = 1.0              # ! note

                    mask_numpy_final = np.logical_and(mask_numpy_final, mask_numpy).astype(np.float32)
                
                visible_np_list.append(mask_numpy)
                flow_np_list.append(flow_numpy)
            
            flow_sq = torch.from_numpy(np.stack(flow_np_list))
            vis_mask_sq = torch.from_numpy(np.stack(visible_np_list)).unsqueeze(-1)
            
            # compute motion_bucket
            motion_bucket_id = torch.mean(torch.sum(flow_sq**2, dim=-1).sqrt())
            
            if np.sum(mask_numpy_final)<=196:           # ori---900 
                raise RuntimeError(f'this video has a bad trajectory..{video.name}')

            # 4、return all dataset values
            sample_dict = {
                "video": frames, 'flow_ori': flow_sq, 'caption': caption, 'fps_id': 8, 'motion_bucket_id': motion_bucket_id,
                "video_name": video.name, "vis_mask_sq": vis_mask_sq, 
            }
            # only do transformation for videos
            if transform is not None:
                sample_dict = transform(sample_dict)
            
            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {_MAX_CONSECUTIVE_FAILURES} retries."
            )

    transformed_videos_dataset = web_videodataset.map(decode_mp4_video_data, handler=wds.handlers.warn_and_continue)

    return transformed_videos_dataset


def VidTar_all_flow(
    data_path: Union[str, List[str]],
    clip_sampler: Dict[str, Any],
    transform: Dict[str, Any] = None,
    decoder: str = "pyav",
    resampled: bool = False,
    data_key: List[str] = ['mp4'],
    remove_folder_list: List[str] = None,
    random_mask_ratio=None,
    use_oss: bool = False,
):
    """
    A helper function to create ``LabeledVideoDataset`` object for the WebVid dataset.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a directory, the directory structure has data_path/0-9/xxxx-xxxx.tar.

        clip_sampler (Dict[str, Any]): The config to define how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        transform (Dict[str, Any]): Config for the callable function that is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        decoder (str): Defines what type of decoder used to decode a video.

    """

    torch._C._log_api_usage_once("dataset.VidTar")

    clip_sampler = instantiate_from_config(clip_sampler)
    transform = instantiate_from_config(transform)
    
    
    return create_video_tar_dataset_all_flow(
        data_path,
        clip_sampler,
        transform,
        decoder,
        resampled,
        data_key,
        remove_folder_list,
        random_mask_ratio=random_mask_ratio,
        use_oss=use_oss,
    )
