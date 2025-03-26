from typing import Callable, Dict, List, Optional, Tuple

import dataset.transforms.functional
import torch
import torchvision.transforms
import cv2
import numpy as np

class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x

class ConvertUint8ToFloat(torch.nn.Module):
    """
    Converts a video from dtype uint8 to dtype float32.
    """

    def __init__(self):
        super().__init__()
        self.convert_func = torchvision.transforms.ConvertImageDtype(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert x.dtype == torch.uint8, "image must have dtype torch.uint8"
        return self.convert_func(x)


class Normalize(torchvision.transforms.Normalize):
    """
    Normalize the (CTHW) video clip by mean subtraction and division by standard deviation

    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        vid = x.permute(1, 0, 2, 3)  # C T H W to T C H W
        vid = super().forward(vid)
        vid = vid.permute(1, 0, 2, 3)  # T C H W to C T H W
        return vid

class RandomShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``dataset.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: str = "bilinear",
        backend: str = "pytorch",
    ):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return dataset.transforms.functional.short_side_scale(
            x, size, self._interpolation, self._backend
        )

class RemoveKey(torch.nn.Module):
    """
    Removes the given key from the input dict. Useful for removing modalities from a
    video clip that aren't needed.
    """

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        if self._key in x:
            del x[self._key]
        return x


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``dataset.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return dataset.transforms.functional.short_side_scale(
            x, self._size, self._interpolation, self._backend
        )


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return dataset.transforms.functional.uniform_temporal_subsample(
            x, self._num_samples, self._temporal_dim
        )

class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``dataset.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return torchvision.transforms.Lambda(
            dataset.transforms.functional.div_255
        )(x)


class WatermarkRemove(torch.nn.Module):
    def __init__(
        self, 
        alpha_file_path: str = './data/webvid/test_data/shutterstock_alpha.png', 
        w_file_path: str = './data/webvid/test_data/shutterstock_W.png'
    ):
        super().__init__()
        self.alpha = cv2.imread(alpha_file_path) / 255.0
        self.W = cv2.imread(w_file_path) / 1.0
        pos = self.alpha.max(axis=-1).nonzero()
        box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
        self.bg = np.zeros(self.W.shape)
        self.bg[box[0]:box[2], box[1]:box[3],:] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        x = dataset.transforms.functional.watermark_remove(
            x.to(torch.uint8), self.alpha, self.W, self.bg
        )
        return x.to(torch.float32)
    
