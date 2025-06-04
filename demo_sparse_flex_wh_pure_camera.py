import sys
sys.dont_write_bytecode = True      # Avoid produce __pycache__ file..


import gradio as gr
import numpy as np
import cv2
from PIL import Image
import uuid
from scipy.interpolate import PchipInterpolator
from vtdm.utils import *
from einops import rearrange, repeat
import torch
from tqdm import tqdm
from torchvision import transforms
from tools.camera import Warper
import warnings
warnings.filterwarnings("ignore")


# CHANGE CKPT PATH--MotionPro-Sparse
ckpt_path = 'checkpoints/MotionPro_Sparse-gs_16k.pt'
base_save_dir = "all_results/test/motionpro_s_pure_camera"
ensure_dirname(base_save_dir)

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

def visualize_drag_v2(background_image_path, splited_tracks, width, height):
    trajectory_maps = []
    
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 5, (255, 0, 0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer

class Drag:
    def __init__(self, device, model_path, cfg_path, model_length):
        self.device = device
        cf = import_filename(cfg_path)
        Net, args = cf.Net, cf.args
        motionpro_net = Net(args)               # Done. checked..
        
        motionpro_net.eval()
        motionpro_net.to(device)

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

        samples = self.motionpro_net.decode_first_stage(samples_z)
        predict_video = rearrange(samples, '(b l) c h w -> b l c h w', b=b)
        all_sample_dict['samples-video'] = predict_video
        save_sample_results(all_sample_dict, output_dir, 'pure_camera_motion', 0, motion_bucket_id=motion_bucket_id, id=id)
        
        outputs['logits_imgs'] = predict_video
        return outputs

    def run(self, first_frame_path, tracking_points, inference_batch_size, motion_bucket_id, first_frame_path_mask=None, img_ratio=None, camera_motion=None):
        original_width, original_height= img_ratio
        self.width, self.height = img_ratio
        mask = None
        # output_dir = os.path.join(base_save_dir, camera_motion)
        output_dir = base_save_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if camera_motion == 'None':
            input_all_points = tracking_points.constructor_args['value']
            resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]

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
                mask_brush = (mask_brush[None,:,:,None]/255).repeat(16,1,1,1).to(torch.int)
            
            if mask is None: 
                mask = torch.zeros(self.model_length, self.height, self.width, 1)
            else:
                mask = mask[None,:,:,None].repeat(16,1,1,1)
            
            if mask_brush is not None:
                mask = torch.logical_or(mask, mask_brush)
        else:
            resized_all_points = []
            warper = Warper(self.height, self.width)
            mask = torch.ones(self.model_length, self.height, self.width, 1)
            first_frame_pil = Image.open(first_frame_path)
            input_drag_dense, visible_mask = warper.get_traj_from_camera_pose(first_frame_pil, camera_motion, num_frames=self.model_length)
            input_drag, motion_bucket_id = warper.process_into_sparse(input_drag_dense, visible_mask)
        
        # Prepare the inputs of network need...
        mask_b = mask.to(input_drag.dtype)                               
        input_drag = torch.concat([input_drag, mask_b], dim=-1).to(self.device)  
        
        dir, base, ext = split_filename(first_frame_path)               # 1.  set trajectory [13, 320, 512, 2]  tensor, not-normed as input..
        id = base.split('_')[-1]
        
        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert('RGB')
        
        visualized_drag, _ = visualize_drag_v2(first_frame_path, resized_all_points, self.width, self.height)
        
        first_frames_transform = transforms.Compose([
                        lambda x: Image.fromarray(x),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        
        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
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

        for i in range(inference_batch_size):
            ouput_tensor = [ouput_video_list[0][i]]
            for j in range(num_inference - 1):
                ouput_tensor.append(ouput_video_list[j+1][i][1:])
            ouput_tensor = torch.cat(ouput_tensor, dim=0)
            outputs_path = os.path.join(output_dir, f'output_{i}_{id}.gif')
            data2file([transforms.ToPILImage('RGB')(utils.make_grid(e.to(torch.float32).cpu(), normalize=True, value_range=(-1, 1))) for e in ouput_tensor], outputs_path,
                      printable=False, duration=1 / 8, override=True)

        return visualized_drag[0], outputs_path


with gr.Blocks() as demo:
    gr.Markdown("""
        <h1 align='center'>MotionPro: A Precise Motion Controller for Image-to-Video Generation</h1>
        <h2 align='center'>CVPR 2025 Paper Demo: Pure Camera Control</h2><br>
    """, unsafe_allow_html=True)


    gr.Markdown(
        """
        ## How to Use MotionPro:<br>
        1. **Upload an Image** - Click to select an image and load it into the input box.<br>
        2. **Brush a Mask** - *(Optional)* When selecting a camera motion type, the model will automatically apply a full mask.<br>
        3. **Initialize Mask & Crop** - Click **"Init mask and crop img"** to process the mask and crop the image.<br>
        4. **Select a Camera Motion Type** - Choose a predefined camera motion type.<br>
        5. **Run Animation** - Click **"Run"** to animate the image based on the selected motion type.<br>
        6. **Reset & Restart** - Click **"Reset"** to clear the image, mask, and motion trajectory paths. You'll need to restart from step 2 or 3.<br>
        """,
        unsafe_allow_html=True
    )

    
    MotionPro_net = Drag("cuda:0", ckpt_path, 'vtdm/motionpro_net.py', 16)
    first_frame_path = gr.State()
    first_frame_path_mask = gr.State()
    tracking_points = gr.State([])
    img_ratio = gr.State()

    def reset_states(first_frame_path, tracking_points):
        first_frame_path = gr.State()
        tracking_points = gr.State([])
    
        return first_frame_path, tracking_points, None

    def preprocess_image(tracking_points, imageMask):
        tracking_points = gr.State([])
        image_pil = imageMask['image']
        raw_w, raw_h = image_pil.size
        
        resize_ratio = max(512/raw_w, 320/raw_h)
        image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        image_pil = transforms.CenterCrop((320, 512))(image_pil)

        id = str(uuid.uuid4())[:4]
        first_frame_path = os.path.join(base_save_dir, 'image_log', 'first_frame', f"first_frame_id_{id}.png")
        os.makedirs(os.path.dirname(first_frame_path), exist_ok=True)
        image_pil.save(first_frame_path)
        
        mask_pil = imageMask['mask']
        mask_pil = mask_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        mask_pil = transforms.CenterCrop((320, 512))(mask_pil)
        first_frame_path_mask = os.path.join(base_save_dir, 'image_log', 'masks', f"first_frame_id_{id}_mask.png")
        os.makedirs(os.path.dirname(first_frame_path_mask), exist_ok=True)
        mask_pil.save(first_frame_path_mask)
        
        return tracking_points, first_frame_path, first_frame_path, first_frame_path_mask

    def preprocess_image_flex_ratio(tracking_points, imageMask):
        tracking_points = gr.State([])
        image_pil = imageMask['image']
        raw_w, raw_h = image_pil.size
        
        image_aspect_ratio = raw_w / raw_h
        
        diff_512_320 = abs(image_aspect_ratio - 512 / 320)
        diff_512_512 = abs(image_aspect_ratio - 512 / 512)
        diff_320_512 = abs(image_aspect_ratio - 320 / 512)
        closest_aspect_ratio = min(diff_512_320, diff_512_512, diff_320_512)
        
        all_ratio = [(512,320), (512,512), (320,512)]
        target_w, target_h = all_ratio[[diff_512_320, diff_512_512, diff_320_512].index(closest_aspect_ratio)]
        
        
        resize_ratio = max(target_w/raw_w, target_h/raw_h)
        image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        image_pil = transforms.CenterCrop((target_h, target_w))(image_pil)

        id = str(uuid.uuid4())[:4]
        first_frame_path = os.path.join(base_save_dir, 'image_log', 'first_frame', f"first_frame_id_{id}.png")
        os.makedirs(os.path.dirname(first_frame_path), exist_ok=True)
        image_pil.save(first_frame_path)
        
        mask_pil = imageMask['mask']
        mask_pil = mask_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        mask_pil = transforms.CenterCrop((target_h, target_w))(mask_pil)
        first_frame_path_mask = os.path.join(base_save_dir, 'image_log', 'masks', f"first_frame_id_{id}_mask.png")
        os.makedirs(os.path.dirname(first_frame_path_mask), exist_ok=True)
        mask_pil.save(first_frame_path_mask)
        
        return tracking_points, first_frame_path, first_frame_path, first_frame_path_mask, (target_w, target_h)

    def add_drag(tracking_points):
        tracking_points.constructor_args['value'].append([])
        return tracking_points
    
    def delete_last_drag(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        
        return tracking_points, trajectory_map
    
    def delete_last_step(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'][-1].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        
        return tracking_points, trajectory_map
    
    def add_tracking_points(tracking_points, first_frame_path, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        tracking_points.constructor_args['value'][-1].append(evt.index)

        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

        return tracking_points, trajectory_map

    
    # with open(os.path.join('assets', 'script.js'), 'r') as f:
    #     js_str = f.read()

    # demo.load(_js=js_str)
    
    # type---{motion_class}-{mode}-{strength}
    camera_motion_list = [
                    'None',
                    'zoom-in-1', 'zoom-in-2', 'zoom-out-1', 'zoom-out-2', 
                    'pan-left-30', 'pan-left-60', 'pan-right-30', 'pan-right-60', 
                    # 'tilt-up-30', 'tilt-up-60', 'tilt-down-30', 'tilt-down-60',         # too large
                    # 'pedestal-up-1', 'pedestal-up-2', 'pedestal-down-1', 'pedestal-down-2', 
                    'truck-left-1', 'truck-left-2', 'truck-right-1', 'truck-right-2', 
                    'roll-clockwise-30', 'roll-clockwise-60', 'roll-anticlockwise-30', 'roll-anticlockwise-60', 
                    'rotate-clockwise-30', 'rotate-clockwise-60', 'rotate-anticlockwise-30', 'rotate-anticlockwise-60',
                    # 'hybrid-in_then_up-default', 'hybrid-out_left_up_down-default',
                    'complex-mode_1-default', 'complex-mode_2-default', 'complex-mode_3-default', 'complex-mode_4-default', 
                    'complex-mode_5-default', 'complex-mode_6-default', 'complex-mode_7-default', 'complex-mode_8-default', 
                    'complex-mode_9-default', 'complex-mode_10-default', 'complex-mode_11-default', 'complex-mode_12-default', 
                    'complex-mode_13-default', 'complex-mode_14-default',
                    ]
    
    with gr.Row():
        with gr.Column(scale=1):
            
            init_drag_button = gr.Button(value="Init mask and crop img")
            # add_drag_button = gr.Button(value="Add Drag")
            reset_button = gr.Button(value="Reset")
            run_button = gr.Button(value="Run")

        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=1):
                    imageMask = gr.ImageMask(label="Input Image", brush_color='	#FF00FF', brush_radius=50.0,
                                                       elem_id="inputmask", type="pil", height=512, width=512,)
                with gr.Column(scale=1):
                    input_image = gr.Image(label=None,
                                        interactive=True,
                                        height=512,
                                        width=512,)
    with gr.Row():
        with gr.Column(scale=1):
            inference_batch_size = gr.Slider(label='Inference Batch Size', 
                                             minimum=1, 
                                             maximum=1, 
                                             step=1, 
                                             value=1)
            
            motion_bucket_id = gr.Slider(label='Motion Bucket',
                                             minimum=1, 
                                             maximum=100, 
                                             step=1, 
                                             value=17)
            camera_motion = gr.Dropdown(choices=camera_motion_list, label='Camera Motion Type')
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=1):
                    output_video = gr.Image(label="Output Video",
                                            height=512,
                                            width=512,)
                with gr.Column(scale=1):
                    output_image = gr.Image(label=None,
                                                    height=512,
                                                    width=512,)



    imageMask.upload(fn=None,
                _js="async function (a) {hr_img = await resize_b64_img(a['image'], 2048); dp_img = await resize_b64_img(hr_img, 1024); return [hr_img, {image: dp_img, mask: null}]}",
                inputs=[imageMask],
                outputs=[input_image, imageMask])           

    init_drag_button.click(preprocess_image_flex_ratio, [tracking_points, imageMask], [tracking_points, input_image, first_frame_path, first_frame_path_mask, img_ratio])
    # add_drag_button.click(add_drag, tracking_points, tracking_points)

    reset_button.click(reset_states, [first_frame_path, tracking_points], [first_frame_path, tracking_points, input_image])

    input_image.select(add_tracking_points, [tracking_points, first_frame_path], [tracking_points, input_image])

    run_button.click(MotionPro_net.run, [first_frame_path, tracking_points, inference_batch_size, motion_bucket_id, first_frame_path_mask, img_ratio, camera_motion], [output_image, output_video])
    
    demo.launch(server_name="0.0.0.0", server_port=7868,debug=True)
