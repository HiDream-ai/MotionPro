import os
import torch
from tqdm import tqdm
import json
from PIL import Image
from vtdm.utils import *
from scipy.interpolate import interp1d, PchipInterpolator
import cv2
import argparse

'''
Eval Fine-grained and object-level motion control based on MC-Bench
'''

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


def save_sample_results_new(images, save_dir=None, batch_idx=0, node_rank=0, motion_bucket_id=None, id=None):
    
    batch_masked_flow = images['flow'].permute(0,1,3,4,2)[:,:,:,:,:2]       # [b,f,h,w,c], torch.tensor
    batch_mask = images['flow'].permute(0,1,3,4,2)[:,:,:,:,-1]
    b,f,h,w = batch_mask.shape
    
    base_save_dir = save_dir
    # filename = "results_gs-{:06}_e-{:06}_b-{:06}".format(0, 0, batch_idx)
    save_video_dir = base_save_dir
    os.makedirs(save_video_dir, exist_ok=True)
    
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
            
            png_save_folder = os.path.join(save_video_dir, 'pngs')
            os.makedirs(png_save_folder, exist_ok=True)
            img_save_path = os.path.join(png_save_folder, f'frame_{j:03d}.png')
            img = Image.fromarray(predict_video[j].detach().cpu().numpy())
            img.save(img_save_path)

        concat_video = torch.stack(video_frame_list)
        video_name = f'sampled_video-motion_bucket_id_{motion_bucket_id:02d}-id_' + id + '.mp4'
        video_path = os.path.join(save_video_dir, video_name)
        
        torchvision.io.write_video(video_path, concat_video, fps=8, video_codec='h264', options={'crf': '10'})
        


class Drag:
    def __init__(self, device, model_path, cfg_path, model_length):
        self.device = device
        cf = import_filename(cfg_path)
        Net, args = cf.Net, cf.args
        motionpro_net = Net(args)               # Done. checked..
        
        motionpro_net.eval()
        motionpro_net.to(device)
        # motionpro_net.half()
        self.motionpro_net = motionpro_net
        
        # 1. load new pt model from svd_flow..
        if model_path.endswith('pt'):
            self.init_from_ckpt(model_path)
        else:
            state_dict = file2data(model_path, map_location='cpu')
            adaptively_load_state_dict(motionpro_net, state_dict)
        
        # self.height = height
        # self.width = width
        # model_step = model_path.split('/')[-2]
        # self.ouput_prefix = f'{model_step}_{width}X{height}'
        self.model_length = model_length
    
    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
        elif path.endswith("pt") or path.endswith("pth"):
            sd_raw = torch.load(path, map_location="cpu")
            sd = {}
            for k in sd_raw['module']:
                sd[k[len('module.'):]] = sd_raw['module'][k]
        else:
            raise NotImplementedError

        # missing, unexpected = self.load_state_dict(sd, strict=True)
        missing, unexpected = self.motionpro_net.load_state_dict(sd, strict=False)            # todo: change into strict true..
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    @torch.no_grad()
    def forward_sample(self, input_drag, input_first_frame, motion_bucket_id, outputs=dict(), output_dir=None, id=None):
        device = self.device
    
        all_sample_dict = {}
        b, l, h, w, c = input_drag.size()
        # drag = self.motionpro_net.apply_gaussian_filter_on_drag(input_drag)
        # drag = torch.cat([torch.zeros_like(drag[:, 0]).unsqueeze(1), drag], dim=1)  # pad the first frame with zero flow
        drag = rearrange(input_drag, 'b l h w c -> b l c h w')
        all_sample_dict['flow'] = drag
        
        input_conditioner = dict()              # *zzw: all is SVD need params..
        input_conditioner['cond_frames_without_noise'] = input_first_frame
        input_conditioner['cond_frames'] = (input_first_frame + 0.02 * torch.randn_like(input_first_frame))
        input_conditioner['motion_bucket_id'] = torch.tensor([motion_bucket_id]).to(drag.device).repeat(b * (l))
        input_conditioner['fps_id'] = torch.tensor([self.motionpro_net.args.fps]).to(drag.device).repeat(b * (l))
        input_conditioner['cond_aug'] = torch.tensor([0.02]).to(drag.device).repeat(b * (l))

        input_conditioner_uc = {}               # copy all
        for key in input_conditioner.keys():
            if key not in input_conditioner_uc and isinstance(input_conditioner[key], torch.Tensor):
                input_conditioner_uc[key] = input_conditioner[key].clone()
        
        c, uc = self.motionpro_net.conditioner.get_unconditional_conditioning(
            input_conditioner,
            batch_uc=input_conditioner_uc,
            force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.motionpro_net.num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...")
            c[k] = repeat(c[k], "b ... -> b t ...", t=self.motionpro_net.num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...")
    
        H, W = input_conditioner['cond_frames_without_noise'].shape[2:]
        shape = (self.motionpro_net.num_frames, 4, H // 8, W // 8)
        randn = torch.randn(shape).to(self.device)

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            2, self.motionpro_net.num_frames
        ).to(self.device)
        additional_model_inputs["num_video_frames"] = self.motionpro_net.num_frames
        additional_model_inputs["flow"] = drag.repeat(2, 1, 1, 1, 1)    # c and uc

        def denoiser(input, sigma, c):
            return self.motionpro_net.denoiser(self.motionpro_net.model, input, sigma, c, **additional_model_inputs)
        
        samples_z = self.motionpro_net.sampler(denoiser, randn, cond=c, uc=uc)
        
        # use vae decoder..---error, this version always flicks.
        # from diffusers import AutoencoderKLTemporalDecoder
        # vae = AutoencoderKLTemporalDecoder.from_pretrained(
        #     '/mnt/afs/zhongwei/subapp/SVD/stable-video-diffusion-img2vid-xt', subfolder="vae", revision=None,
        #     ).to(device=samples_z.device)
        # from test_utils import decode_latents
        # new_samples = decode_latents(vae, samples_z, num_frames=self.model_length, decode_chunk_size=1)   # todo: to save GPU memory check for comparison, the same diff in gif show.
        # outputs['logits_imgs'] = new_samples.permute(0,2,1,3,4)
        
        samples = self.motionpro_net.decode_first_stage(samples_z)
        predict_video = rearrange(samples, '(b l) c h w -> b l c h w', b=b)
        all_sample_dict['samples-video'] = predict_video
        save_sample_results_new(all_sample_dict, output_dir, 0, 0, motion_bucket_id=motion_bucket_id, id=id)
        
        outputs['logits_imgs'] = predict_video
        return outputs

    def run(self, first_frame_path, resized_all_points, inference_batch_size, motion_bucket_id, first_frame_path_mask=None, img_ratio=None, image_label_name=None, output_dir=None):
        original_width, original_height= img_ratio
        self.width, self.height = img_ratio
        mask = None
        
        input_drag = torch.zeros(self.model_length, self.height, self.width, 2)
        for splited_track in resized_all_points:
            if len(splited_track) == 1: # stationary point
                displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[:self.model_length]
            if len(splited_track) < self.model_length:
                splited_track = splited_track + [splited_track[-1]] * (self.model_length -len(splited_track))
            for i in range(self.model_length):
                start_point = splited_track[0]
                end_point = splited_track[i]
                start_y, start_x = int(start_point[1]), int(start_point[0])
                input_drag[i,start_y-4:start_y+4, start_x-4:start_x+4, 0] = end_point[0] - start_point[0]          # change input area values
                input_drag[i,start_y-4:start_y+4, start_x-4:start_x+4, 1] = end_point[1] - start_point[1]
                mask = (torch.sqrt(torch.sum(input_drag[i] ** 2, dim=-1)) > 0.0).to(torch.int)
        
        # new type motion brush..
        assert first_frame_path_mask is not None, 'impossible..'
        if first_frame_path_mask is not None: 
            mask_brush = image2arr(first_frame_path_mask)
            mask_brush = torch.from_numpy(mask_brush)
            assert torch.equal(mask_brush[:,:,0], mask_brush[:,:,2]), 'error, this mask should be the same in diff channels'
            mask_brush = torch.any(mask_brush != 0, dim=-1).type(torch.uint8) * 255
            mask_brush = (mask_brush[None,:,:,None]/255).repeat(16,1,1,1).to(torch.int)       # todo: change..
            
            # if torch.sum(mask_brush)==0:
            #      mask_brush = torch.ones(self.model_length, self.height, self.width, 1)     # ! no use...
            
        
        if mask is None: 
            mask = torch.zeros(self.model_length, self.height, self.width, 1)
        else:
            mask = mask[None,:,:,None].repeat(16,1,1,1)
        
        if mask_brush is not None:
            mask = torch.logical_or(mask, mask_brush)
        
        input_drag = torch.concat([input_drag, mask], dim=-1)           # todo_for add frame1: change shape [b,h,w,3] -> [b,h,w,6]
        
        dir, base, ext = split_filename(first_frame_path)               # 1.  set trajectory [13, 320, 512, 2]  tensor, not-normed as input..
        id = base.split('_')[-1]
        
        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert('RGB')
        
        first_frames_transform = transforms.Compose([
                        lambda x: Image.fromarray(x),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        
        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in range(num_inference):
            if not outputs:
                first_frames = image2arr(first_frame_path)              # 2.  read first_frame as [1, 3, 320, 512] tensor, value range [-1,1]
                first_frames = repeat(first_frames_transform(first_frames), 'c h w -> b c h w', b=inference_batch_size).to(self.device)
            else:
                first_frames = outputs['logits_imgs'][:, -1]
            
            outputs = self.forward_sample(                      # *zzw: input_drag: [13,h,w,2]
                                            repeat(input_drag[i*self.model_length:(i+1)*self.model_length], 'l h w c -> b l h w c', b=inference_batch_size).to(self.device), 
                                            first_frames,
                                            motion_bucket_id, output_dir=output_dir, id=id)           # 3.  default value..for inference
            ouput_video_list.append(outputs['logits_imgs'])


def load_json(file_path, key=None):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    if key is not None:
        return data[key]
    return data


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/MotionPro_Sparse-gs_16k.pt', help="ckpt path")
    parser.add_argument("--dataset_path", type=str, 
                        default='/mnt/zhongwei/zhongwei/all_good_tools/metrics/our_bench/MC-Bench', 
                        help="dataset path")
    parser.add_argument("--output_dir", type=str, default='all_results/eval/mc_bench', help="output path")
    parser.add_argument("--seed", type=int, default=2025, help="test sampling seed..")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="inference_batch_size")
    parser.add_argument("--motion_bucket_id", type=int, default=17, help="motion_bucket_id")
    
    return parser


def main():

    # --------------------1、read config
    parser = get_parser()
    args = parser.parse_args()
    inference_batch_size = args.inference_batch_size
    motion_bucket_id = args.motion_bucket_id
    set_seed(args.seed)

    base_output_dir = args.output_dir
    if args.ckpt_path is not None:
        extra_name = args.ckpt_path.split('/')[-2]
        base_output_dir = os.path.join(base_output_dir, extra_name)
    
    os.makedirs(base_output_dir,exist_ok=True)

    MotionPro_net = Drag("cuda:0", args.ckpt_path, 'vtdm/motionpro_net.py', 16)
    
    base_folder = sorted(os.listdir(args.dataset_path))
    for folder in base_folder:
        all_sub_folder = sorted(os.listdir(os.path.join(args.dataset_path, folder)))
        
        for sub_folder in tqdm(all_sub_folder):
            data_path = os.path.join(args.dataset_path, folder, sub_folder)
            
            first_frame_path = os.path.join(data_path, 'first_frame.png')
            first_frame_path_mask = os.path.join(data_path, 'first_frame_mask.png')
            
            img_pil = Image.open(first_frame_path)
            img_ratio = img_pil.size
            image_label_name = sub_folder
            output_dir = os.path.join(base_output_dir, folder, sub_folder)
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(first_frame_path, os.path.join(output_dir, 'first_frame.png'))
            shutil.copy(os.path.join(data_path, 'user_drag.png'), os.path.join(output_dir, 'user_drag.png'))
            
            resized_all_points = load_json(os.path.join(data_path, 'meta_data.json'), key='traj_key_points')
            MotionPro_net.run(first_frame_path, resized_all_points, inference_batch_size, motion_bucket_id, first_frame_path_mask, 
                             img_ratio, image_label_name, output_dir)
            

if __name__ == "__main__":
    main()      
