import torch
from typing import Any, Dict, List

from einops import rearrange, repeat
from sgm.models.diffusion import DiffusionEngine
from vtdm.utils import instantiate_from_config
from safetensors.torch import load_file as load_safetensors



class VideoLDM(DiffusionEngine):
    def __init__(self, num_samples, trained_param_keys=[''], *args, **kwargs):
        self.trained_param_keys = trained_param_keys
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

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
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    @torch.no_grad()
    def add_custom_cond(self, batch, infer=False):
        batch['num_video_frames'] = self.num_samples
        
        image = batch['video'][:, :, 0]
        batch['cond_frames_without_noise'] = image.half()
        
        N = batch['video'].shape[0]
        if not infer:
            cond_aug = ((-3.0) + (0.5) * torch.randn((N,))).exp().cuda().half()
        else:
            cond_aug = torch.full((N, ), 0.02).cuda().half()
        batch['cond_aug'] = cond_aug
        batch['cond_frames'] = (image + rearrange(cond_aug, 'b -> b 1 1 1') * torch.randn_like(image)).half()
            
        # for dataset without indicator
        if not 'image_only_indicator' in batch:
            batch['image_only_indicator'] = torch.zeros((N, self.num_samples)).cuda().half()
        
        if "flow_ori" in batch and 'flow_brush_mask' in batch and batch['flow_brush_mask'] is not None:      # MotionPro-Sparse: input 3 channel for optical flow as condition..
            batch["flow"] = torch.concat([batch['flow_ori'].permute(0,1,4,2,3), batch['flow_brush_mask'].permute(0,1,4,2,3)], dim=2)    
        elif "flow_ori" in batch and "vis_mask_sq" in batch:        # MotionPro-Dense
            batch["flow"] = torch.concat([batch['flow_ori'].permute(0,1,4,2,3), batch['vis_mask_sq'].permute(0,1,4,2,3)], dim=2)    
        else:
            raise ValueError("No optical flow or mask map provided")    
        
        return batch

    def shared_step(self, batch: Dict) -> Any:
        frames = self.get_input(batch) # b c t h w
        batch = self.add_custom_cond(batch)    # for dragnvwa_svd:
        
        frames_reshape = rearrange(frames, 'b c t h w -> (b t) c h w')
        x = self.encode_first_stage(frames_reshape)
        
        batch["global_step"] = self.global_step
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        frames = self.get_input(batch)                          #  ori_video
        batch = self.add_custom_cond(batch, infer=True)
        N = min(frames.shape[0], N)
        frames = frames[:N]
        x = rearrange(frames, 'b c t h w -> (b t) c h w')
        
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )
        # fix condition
        for k in ["crossattn", "concat", "vector"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.num_samples)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...")
            c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_samples)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...")
            assert torch.equal(c[k][0], c[k][0]), 'no possible, repeat error..'
        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = batch['image_only_indicator'].repeat(2,1)
        additional_model_inputs["num_video_frames"] = self.num_samples
        additional_model_inputs["flow"] = batch["flow"].repeat(2, 1, 1, 1, 1)    # c and uc
        
        # 1. log ori_videos and optical flow and mask map..
        x = x.to(self.device)
        z = self.encode_first_stage(x.half())
        x_rec = self.decode_first_stage(z.half())
        log["reconstructions-video"] = rearrange(x_rec, '(b t) c h w -> b c t h w', t=self.num_samples)
        log['flow'] = batch['flow']
        
        # begin sample predict videos
        def denoiser(input, sigma, c):
            return self.denoiser(
                self.model, input, sigma, c, **additional_model_inputs
            )
            
        if sample:
            with self.ema_scope("Plotting"):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    randn = torch.randn(z.shape, device=self.device)
                    samples = self.sampler(denoiser, randn, cond=c, uc=uc)
            samples = self.decode_first_stage(samples.half())
            
            # 2. log predict videos
            log["samples-video"] = rearrange(samples, '(b t) c h w -> b c t h w', t=self.num_samples)
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        # params = list(self.model.parameters())
        names = []
        params = []
        for name, param in self.model.named_parameters():
            flag = False
            for k in self.trained_param_keys:
                if k in name:
                    names += [name]
                    param.requires_grad = True
                    params += [param]
                    flag = True
                if flag:
                    break
        
        print(names)
        
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
                
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt
