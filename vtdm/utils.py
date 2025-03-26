import importlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import torch
import torchvision
import imageio
import cv2
from typing import List
from einops import rearrange
import logging
logging.basicConfig(level=logging.WARNING)  

import colorlog
import shutil
import sys
import yaml
from torchvision.utils import flow_to_image
from torchvision import utils


def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'), loop=0)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('Do not support this type')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def image2pil(filename):
    return Image.open(filename)

def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], -1) for e in pil])
    else:
        arr = np.array(pil)
    return arr

def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)

def save_sample_results(images, base_save_dir=None, base_name=None, node_rank=0, motion_bucket_id=None, id=None):
    
    batch_masked_flow = images['flow'].permute(0,1,3,4,2)[:,:,:,:,:2]
    batch_mask = images['flow'].permute(0,1,3,4,2)[:,:,:,:,-1]
    b,f,h,w = batch_mask.shape
    
    os.makedirs(base_save_dir, exist_ok=True)
    
    for i in range(b):
        video_frame_list = []
        predict_video = torch.clamp(images['samples-video'].permute(0,1,3,4,2)[i], -1., 1.)/2 + 0.5
        predict_video = (predict_video * 255).to(torch.uint8)           # [f,h,w,c], torch.uint8
        
        for j in range(f):
            frame1 = predict_video[0].detach().cpu()
            masked_flow = flow_to_image(batch_masked_flow[i,j].to(torch.float32).unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze().detach().cpu()
            mask = (batch_mask[i,j]*255).to(torch.uint8).unsqueeze(-1).repeat(1,1,3).detach().cpu()
            
            masked_arrow = torch.from_numpy(
                attach_arrows(batch_masked_flow[i,j].detach().cpu().numpy(), base_save_dir, 0, node_rank, idx=j)
                ).to(torch.uint8)
            
            video_frame_list.append(torch.concat([frame1, mask, masked_flow, masked_arrow, predict_video[j].detach().cpu()], dim=1))

        concat_video = torch.stack(video_frame_list)
        video_name = f'{base_name}-id_' + id + '.mp4'
        video_path = os.path.join(base_save_dir, video_name)
        torchvision.io.write_video(video_path, concat_video, fps=8, video_codec='h264', options={'crf': '10'})
        
        video_name = f'{base_name}-id_' + id + '-single.mp4'
        video_path = os.path.join(base_save_dir, video_name)
        torchvision.io.write_video(video_path, concat_video[:, :, -w:, :], fps=8, video_codec='h264', options={'crf': '10'})


def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    try:
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    except Exception as e:
        logger.warning('load error %s', e)
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        logger.warning('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)             # update some model ckpts..

    missing_keys = [k for k in target_dict.keys() if k not in common_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        logger.warning(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        logger.warning(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0:
        logger.warning("Strictly Loaded state_dict.")

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    
    if extname in ['pth', 'ckpt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            logger.info('Loaded data from %s' % os.path.abspath(filename))
    return data


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_logger(filename=None):
    """
    examples:
        logger = get_logger('try_logging.txt')

        logger.debug("Do something.")
        logger.info("Start print log.")
        logger.warning("Something maybe fail.")
        try:
            raise ValueError()
        except ValueError:
            logger.error("Error", exc_info=True)

        tips:
        DO NOT logger.inf(some big tensors since color may not helpful.)
    """
    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = torch.distributed.get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger



logger = get_logger()


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def attach_arrows(flow, save_path=None, iteration=None, process=None, idx=None,
                  color="r", angles='xy',scale_units='xy', scale=1., **quiver_kwargs):

    height, width, _ = flow.shape

    norm = np.linalg.norm(flow, axis=-1)
    y_grid, x_grid = np.nonzero(norm)

    plt.quiver(
        x_grid,
        y_grid,
        flow[y_grid, x_grid, 0],
        flow[y_grid, x_grid, 1],
        color=color,
        angles=angles, scale_units=scale_units, scale=scale, 
        **quiver_kwargs,
    )
    # plt.gca().invert_xaxis()
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.autoscale(False)
    plt.gca().invert_yaxis()
    cache_dir = os.path.join(save_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'cache_iter_{iteration:02d}_{process:02d}_{idx:02d}.png')
    plt.savefig(cache_path, format='png')
    plt.close()

    img = Image.open(cache_path)
    new_width, new_height = img.size
    ratio = new_width / new_height
    img = np.array(img.resize((int(ratio * height), height)).convert("RGB"))
    
    os.remove(cache_path)
    del flow
    return img   


def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    video = video.mul_(std).add_(mean)  # unnormalize back to [0,1]
    video.clamp_(0, 1)
    images = rearrange(video, 'i c f h w -> (i f) h w c')
    images = images.unbind(dim=0)
    images = [(image.cpu().numpy() * 255).astype('uint8') for image in images]  # f h w c
    return images


def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None, save_to_gif=False, use_cv2=False, fps=8) -> str:
    h, w, c = video_frames[0].shape
    if save_to_gif:
        image_lst = []
        if output_video_path.endswith('mp4'):
            output_video_path = output_video_path[:-3] + 'gif'
        for i in range(len(video_frames)):
            image_lst.append(video_frames[i])
        imageio.mimsave(output_video_path, image_lst, fps=fps)     
        return output_video_path
    else:
        if use_cv2:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
            for i in range(len(video_frames)):
                img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
                video_writer.write(img)
            video_writer.release()
        else:
            duration = math.ceil(len(video_frames) / fps)
            append_num = duration * fps - len(video_frames)
            for k in range(append_num): video_frames.append(video_frames[-1])
            video_stack = np.stack(video_frames, axis=0)
            video_tensor = torch.from_numpy(video_stack)
            torchvision.io.write_video(output_video_path, video_tensor, fps=fps, options={"crf": "17"})
        return output_video_path