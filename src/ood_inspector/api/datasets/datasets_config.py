import dataclasses
from typing import Any, Dict, Optional, Tuple

import omegaconf

from ood_inspector.api import augmentation_config, transform_config


@dataclasses.dataclass
class TransformationStackConfig:
    transformation: transform_config.TransformConfig = omegaconf.MISSING
    augmenter: Optional[augmentation_config.AugmenterConfig] = None
    _target_: str = "ood_inspector.datasets.dataset.TransformationStack"


@dataclasses.dataclass
class EvaluationTransformationStackConfig(TransformationStackConfig):
    transformation: Any = transform_config.EvaluationTransformConfig


@dataclasses.dataclass
class AdaptationTransformationStackConfig(TransformationStackConfig):
    transformation: Any = transform_config.AdaptationTransformConfig


@dataclasses.dataclass
class DatasetConfig:
    """Any torch Dataset can be passed to a InspectorDataset"""


@dataclasses.dataclass
class InspectorDatasetConfig:
    _target_: str = "ood_inspector.datasets.dataset.InspectorDataset"
    dataset: DatasetConfig = omegaconf.MISSING
    number_of_classes_per_attribute: Dict[str, int] = omegaconf.MISSING
    default_attribute: str = "label_"
    transformations: Optional[TransformationStackConfig] = None
    input_size: Optional[Tuple[int, int, int]] = None
    input_mean: Optional[Tuple[float, float, float]] = None
    input_std: Optional[Tuple[float, float, float]] = None
