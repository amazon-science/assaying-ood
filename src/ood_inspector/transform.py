import math
from typing import Tuple

from torchvision.transforms import transforms as tv_transforms


def load_base_transform(
    input_size: Tuple[int, int, int],
    input_mean: Tuple[float, float, float],
    input_std: Tuple[float, float, float],
    training: bool = False,
) -> tv_transforms.Compose:
    """Creates a set of basic transforms to be applied on the input of vision models."""

    input_size = input_size[-2:]  # drop number of channels

    if training:
        return tv_transforms.Compose(
            [
                tv_transforms.RandomResizedCrop(input_size),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=input_mean, std=input_std),
            ]
        )

    # Compute image size for the Resize before CenterCrop (typically=256)
    crop_percentage = 224 / 256
    assert len(input_size) == 2
    if input_size[-1] == input_size[-2]:
        size_before_crop = int(math.floor(input_size[0] / crop_percentage))
    else:
        size_before_crop = tuple([int(x / crop_percentage) for x in input_size])

    return tv_transforms.Compose(
        [
            tv_transforms.Resize(size_before_crop),
            tv_transforms.CenterCrop(input_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=input_mean, std=input_std),
        ]
    )
