# Need refine for more flexible transformer

import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import hydra
from torchvision.transforms import Compose


def build_transforms(transforms_config: Iterable[Mapping[str, Any]]) -> Compose:
    """
    A utility function to build data transforsm from a list of Hydra/Omega Conf
    objects. This utility method is called by
    `pytorchvideo_trainer.datamodule.PyTorchVideoDataModule` class to build a
    sequence of transforms applied during each phase(train, val and test).

    Uses torchvision.transforms.Compose to build a seuquence of transforms.

    Examples of config objects used by this method can be found in,
    `pytorchvide_trainer/conf/datamodule/transforms/`

    Args:
        transforms_config: A list of hydra config objects wherein, each element
        represents config associated with a single transforms.

        An example of this would be,
        ```
        - _target_: pytorchvideo.transforms.ApplyTransformToKey
            transform:
            - _target_: pytorchvideo.transforms.UniformTemporalSubsample
                num_samples: 16
            - _target_: pytorchvideo.transforms.Div255
            - _target_: pytorchvideo.transforms.Normalize
                mean: [0.45, 0.45, 0.45]
                std: [0.225, 0.225, 0.225]
            - _target_: pytorchvideo.transforms.ShortSideScale
                size: 224
            key: video
        - _target_: pytorchvideo.transforms.UniformCropVideo
            size: 224
        - _target_: pytorchvideo.transforms.RemoveKey
            key: audio
        ```
    """
    transform_list = [build_single_transform(config) for config in transforms_config]
    transform = Compose(transform_list)
    return transform


def build_single_transform(config: Mapping[str, Any]) -> Callable[..., object]:
    """
    A utility method to build a single transform from hydra / omega conf objects.

    If the key "transform" is present in the give config, it recursively builds
    and composes transforms using  the `torchvision.transforms.Compose` method.
    """
    config = dict(config)
    if "transform" in config:
        assert isinstance(config["transform"], Sequence)
        transform_list = [
            build_single_transform(transform) for transform in config["transform"]
        ]
        transform = Compose(transform_list)
        config.pop("transform")
        return hydra.utils.instantiate(config, transform=transform)
    return hydra.utils.instantiate(config)
