import abc
import copy
import enum
import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional

import imagenet_c
import numpy as np
import torch
from torchvision.transforms import transforms as tv_transforms

from ood_inspector.datasets.dataset import ChainedDataset, InspectorDataset


class ImageNetCCorruption(enum.Enum):
    gaussian_noise = 0
    shot_noise = 1
    impulse_noise = 2
    defocus_blur = 3
    # We don't use glass_blur since it is very slow, and can lead to crashes.
    # glass_blur = 4
    motion_blur = 5
    zoom_blur = 6
    snow = 7
    # Corruption 8 (frost) is not supported when imagenet_c package is installed using pip,
    # see https=//github.com/hendrycks/robustness/issues/4#issuecomment-427226016.
    # frost = 8
    fog = 9
    brightness = 10
    contrast = 11
    elastic_transform = 12
    pixelate = 13
    jpeg_compression = 14
    speckle_noise = 15
    gaussian_blur = 16
    spatter = 17
    saturate = 18


def _combine_datasets(datasets: List[Any]) -> Any:
    """Combine list of datasets into one dataset."""
    if not datasets:
        return datasets
    all_same_type = all([isinstance(dataset, type(datasets[0])) for dataset in datasets])
    if not all_same_type:
        raise ValueError("All elements in `datasets` have to be of the same type.")

    if isinstance(datasets[0], InspectorDataset):
        return ChainedDataset(datasets)
    return torch.utils.data.ConcatDataset(datasets)


class Corruption(metaclass=abc.ABCMeta):
    def __init__(
        self,
        datasets: Optional[Dict[str, Any]] = None,
        corruption_types: Optional[List[enum.Enum]] = None,
        corruption_severities: Optional[List[int]] = None,
        combine_corruption_types: bool = False,
    ):
        self.datasets = datasets
        self.corruption_types = corruption_types
        self.corruption_severities = corruption_severities
        self.combine_corruption_types = combine_corruption_types

    """Makes corrupted datasets.

        Args:
            datasets: datasets to which corruption should be applied.
            corruption_types: which corruption types to apply.
            corruption_severities: corruption severities (intensity).
            combine_corruption_types: If true, corrupted datasets for each severity level are
                combined. Else, all corrupted datasets are returned indivually.
        """

    @abc.abstractmethod
    def _corrupt_image(
        self, image: torch.Tensor, corruption_type: enum.Enum, severity: int
    ) -> torch.Tensor:
        """Apply corruption to image.

        Args:
            image (torch.Tensor): image tensor with values in [0, 1] and shape (3, 224, 224).
            corruption_type (Enum): which corruption to apply.
            severity (int): corruption intensity (1 to 5).

        Returns:
            torch.Tensor: corrupted version of given image.
        """

    def _corrupt_dataset(self, dataset: InspectorDataset):
        """Apply corruptions to dataset.

        Args:
            dataset: dataset to which corruption should be applied.

        Returns:
            Dict of datasets that include all corrupted inputs for each severity level.
        """
        # Make corrupted dataset for each corruption type and severity level.
        corrupted_datasets = defaultdict(list)
        for severity in self.corruption_severities:
            for corruption_type in self.corruption_types:
                corruption_transform = functools.partial(
                    self._corrupt_image, corruption_type=corruption_type, severity=severity
                )
                corrupted_dataset = copy.deepcopy(dataset)
                corrupted_dataset.transformations.pre_normalization = tv_transforms.Compose(
                    [
                        corrupted_dataset.transformations.pre_normalization,
                        corruption_transform,
                    ]
                )
                if self.combine_corruption_types:
                    corrupted_datasets[f"severity_{severity}"].append(corrupted_dataset)
                else:
                    corrupted_datasets[
                        f"{corruption_type.name}_severity_{severity}"
                    ] = corrupted_dataset

            if self.combine_corruption_types:
                corrupted_datasets[f"severity_{severity}"] = _combine_datasets(
                    corrupted_datasets[f"severity_{severity}"]
                )

        return corrupted_datasets

    def apply_corruptions_to_datasets(self) -> Dict[str, Any]:
        """Corrupt datasets.

        Returns:
            Dict[str, Any]: Corrupted versions of the datasets for each severity level.
        """
        all_datasets = dict()
        for dataset_name, dataset in self.datasets.items():
            corrupted_datasets = self._corrupt_dataset(dataset)
            for corruption_key, corrupted_dataset in corrupted_datasets.items():
                corrupted_name = f"Corrupted_{dataset_name}_{corruption_key}"
                if corrupted_name in self.datasets.keys():
                    raise ValueError(
                        f"Could not add corrupted dataset with name `{corrupted_name}` "
                        "since the name is already present in the provided datasets dictionary."
                    )
                all_datasets[corrupted_name] = corrupted_dataset

        return all_datasets


class NoCorruption(Corruption):
    def _corrupt_image(
        self, image: torch.Tensor, corruption_type: enum.Enum, severity: int
    ) -> torch.Tensor:
        del corruption_type
        del severity
        return image

    def apply_corruptions_to_datasets(self, *args, **kwargs) -> Dict[str, Any]:
        return dict()


class ImageNetCTypeCorruption(Corruption):
    """TODO(armannic): add docstring"""

    def __init__(
        self,
        datasets: Optional[Dict[str, Any]] = None,
        corruption_types: Optional[List[ImageNetCCorruption]] = None,
        corruption_severities: Optional[List[int]] = None,
        combine_corruption_types: bool = False,
    ):
        # Use all available corruptions if None is provided.
        self.datasets = datasets
        if corruption_types is None:
            self.corruption_types = list(ImageNetCCorruption)
        else:
            self.corruption_types = corruption_types
        if corruption_severities is None:
            self.corruption_severities = [severity for severity in range(1, 6)]
        else:
            self.corruption_severities = corruption_severities
        self.combine_corruption_types = combine_corruption_types

    def _corrupt_image(
        self, image: torch.Tensor, corruption_type: ImageNetCCorruption, severity: int
    ) -> torch.Tensor:
        """Wrapper around imagenet_c.corrupt."""
        if severity not in range(1, 6):
            raise ValueError(f"Severity has to be beteween 1 and 5, but {severity} given.")
        if image.shape != (3, 224, 224):
            raise ValueError(f"Image shape has to be (3, 224, 224), but {image.shape} given.")

        # Transform image to the format expected by imagenet_c.
        image = 255 * image
        image = image.permute(1, 2, 0)
        image = image.numpy().astype(np.uint8)

        # Apply corruption.
        image = imagenet_c.corrupt(image, corruption_name=corruption_type.name, severity=severity)

        # Transform back to the pytorch format.
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1)
        image = image / 255.0

        return image
