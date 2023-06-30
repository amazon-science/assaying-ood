import dataclasses
from typing import Any, Dict, List, Optional

import hydra.core.config_store as hydra_config_store

from ood_inspector.api.datasets import datasets_config
from ood_inspector.corruption import ImageNetCCorruption


@dataclasses.dataclass
class CorruptionConfig:
    corruption_severities: Optional[List[int]] = None
    corruption_types: Optional[List[Any]] = None
    combine_corruption_types: bool = False
    datasets: Optional[Dict[str, datasets_config.InspectorDatasetConfig]] = None


@dataclasses.dataclass
class NoCorruptionConfig(CorruptionConfig):
    _target_: str = "ood_inspector.corruption.NoCorruption"


@dataclasses.dataclass
class ImageNetCTypeCorruptionConfig(CorruptionConfig):
    corruption_types: Optional[List[ImageNetCCorruption]] = None
    _target_: str = "ood_inspector.corruption.ImageNetCTypeCorruption"


config_store = hydra_config_store.ConfigStore.instance()
config_store.store(group="corruption", name="no_corruption", node=NoCorruptionConfig)
config_store.store(group="corruption", name="imagenet_c_type", node=ImageNetCTypeCorruptionConfig)
