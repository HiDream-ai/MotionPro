import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)
from vtdm.lora import inject_trainable_lora, inject_trainable_lora_extended, extract_lora_ups_down, _find_modules, fuse_lora_all_linear
import copy

def inject_lora(use_lora, model, replace_modules, is_extended=False, dropout=0.0, r=16):
    injector = (
        inject_trainable_lora if not is_extended
        else
        inject_trainable_lora_extended
    )

    params = None
    negation = None

    if use_lora:
        REPLACE_MODULES = replace_modules
        injector_args = {
            "model": model,
            "target_replace_module": REPLACE_MODULES,
            "r": r
        }
        if not is_extended: injector_args['dropout_p'] = dropout

        params, negation = injector(**injector_args)
        for _up, _down in extract_lora_ups_down(
                model,
                target_replace_module=REPLACE_MODULES):

            if all(x is not None for x in [_up, _down]):
                print(f"Lora successfully injected into {model.__class__.__name__}.")

            break

    return params, negation


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        unet_lora_rank: int = 32,
        unet_lora_rank_new: Optional[int] = None,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)             # DONE: change flow network conv_in into 3 channel..
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)                 
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config                      # lr scheduler
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        
        # ----------special set for lora ablation_new...
        # if ckpt_path is not None and ckpt_path.endswith('.safetensors'):
        #     from safetensors.torch import load_file
        #     model_weights = load_file(ckpt_path)
        #     missing_keys, unexpected_keys = self.load_state_dict(model_weights, strict=False)
        #     print(f'missing: {missing_keys}')
        #     print(f'unexpected: {unexpected_keys}')

        # if unet_lora_rank_new is None or unet_lora_rank_new !=0:
        #     unet_lora_params, unet_negation = inject_lora(
        #                 True, self, ['OpenAIWrapper'], is_extended=False, r=unet_lora_rank_new
        #             )
        
        # ------------Ori version
        if unet_lora_rank_new is None or unet_lora_rank_new !=0:
            unet_lora_params, unet_negation = inject_lora(
                    True, self, ['OpenAIWrapper'], is_extended=False, r=unet_lora_rank
                )
        
        if ckpt_path is not None:
            if ckpt_path.endswith('pth'):
                data = torch.load(ckpt_path, map_location="cpu")
                if unet_lora_rank_new is not None and unet_lora_rank_new==0:
                    self.adaptively_load_state_dict_no_lora(data)
                else:
                    self.adaptively_load_state_dict(data)   # change..
            else:
                self.init_from_ckpt(ckpt_path)
        
        # # Merge lora_params, and delete lora layers
        # # ref: https://github.com/huggingface/diffusers/blob/047bf492914ddc9393070b8f73bba5ad5823eb29/src/diffusers/models/lora.py#L119
        # if unet_lora_rank_new is not None and unet_lora_rank_new !=0:
        #     fuse_lora_all_linear(self, ['OpenAIWrapper'])
        
        #     unet_lora_params, unet_negation = inject_lora(
        #             True, self, ['OpenAIWrapper'], is_extended=False, r=unet_lora_rank_new
        #         )
    
    
    def adaptively_load_state_dict_no_lora(self, state_dict):
        target_dict = self.state_dict()

        try:
            common_dict = {}
            state_dict_new = {}
            for k,v in state_dict.items():
                if 'linear' in k:
                    k = k.replace('.linear', '')
                state_dict_new[k] = v
                if k in target_dict and v.size() == target_dict[k].size():
                    common_dict[k] = v
                elif 'flow_blocks' in k:
                    print(f'process flow_net conv_in model params {k}')
                    v_copy = copy.deepcopy(v) * 0.0
                    v_new = torch.concat([v, v_copy[:,:1,:,:]], dim=1)
                    common_dict[k] = v_new
                else:
                    pass
        except Exception as e:
            print('load error %s', e)
            common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

        if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
                self.state_dict()['param_groups'][0]['params']:
            print('Detected mismatch params, auto adapte state_dict to current')
            common_dict['param_groups'][0]['params'] = self.state_dict()['param_groups'][0]['params']
        target_dict.update(common_dict)
        self.load_state_dict(target_dict)             # update some model ckpts..

        missing_keys = [k for k in target_dict.keys() if k not in common_dict]
        unexpected_keys = [k for k in state_dict_new.keys() if k not in common_dict]

        if len(unexpected_keys) != 0:
            print(
                f"Some weights of state_dict were not used in target: {unexpected_keys}"
            )
        if len(missing_keys) != 0:
            print(
                f"Some weights of state_dict are missing used in target {missing_keys}"
            )
        if len(unexpected_keys) == 0 and len(missing_keys) == 0:
            print("Strictly Loaded state_dict.")    
            
    def adaptively_load_state_dict(self, state_dict):
        target_dict = self.state_dict()

        try:
            common_dict = {}
            for k,v in state_dict.items():
                if k in target_dict and v.size() == target_dict[k].size():
                    common_dict[k] = v
                elif 'flow_blocks' in k:
                    print(f'process flow_net conv_in model params {k}')
                    v_copy = copy.deepcopy(v) * 0.0
                    v_new = torch.concat([v, v_copy[:,:1,:,:]], dim=1)
                    common_dict[k] = v_new
                else:
                    pass
        except Exception as e:
            print('load error %s', e)
            common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

        if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
                self.state_dict()['param_groups'][0]['params']:
            print('Detected mismatch params, auto adapte state_dict to current')
            common_dict['param_groups'][0]['params'] = self.state_dict()['param_groups'][0]['params']
        target_dict.update(common_dict)
        self.load_state_dict(target_dict)             # update some model ckpts..

        missing_keys = [k for k in target_dict.keys() if k not in common_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

        if len(unexpected_keys) != 0:
            print(
                f"Some weights of state_dict were not used in target: {unexpected_keys}"
            )
        if len(missing_keys) != 0:
            print(
                f"Some weights of state_dict are missing used in target {missing_keys}"
            )
        if len(unexpected_keys) == 0 and len(missing_keys) == 0:
            print("Strictly Loaded state_dict.")

    
    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
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
            
        if len(unexpected) == 0 and len(missing) == 0:
            print("Strictly Loaded state_dict.")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
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

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

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

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log
