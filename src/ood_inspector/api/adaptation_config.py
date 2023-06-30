import dataclasses
from typing import Any, List, Optional

import hydra.core.config_store as hydra_config_store
import omegaconf

from ood_inspector.api import torch_core_config
from ood_inspector.api.datasets import datasets_config

_FINE_TUNE_DEFAULTS = [
    "_self_",
    {"lr_scheduler": None},
]


@dataclasses.dataclass
class AdaptationConfig:
    pass


@dataclasses.dataclass
class NoAdaptationConfig(AdaptationConfig):
    _target_: str = "ood_inspector.adaptation.NoAdaptation"


@dataclasses.dataclass
class FineTuneConfig(AdaptationConfig):
    _target_: str = "ood_inspector.adaptation.FineTune"
    defaults: List[Any] = dataclasses.field(default_factory=lambda: _FINE_TUNE_DEFAULTS)

    dataset: datasets_config.InspectorDatasetConfig = omegaconf.MISSING
    lr_scheduler: Optional[torch_core_config.LRSchedulerConfig] = None
    dataloader: torch_core_config.DataLoaderConfig = torch_core_config.TorchDataLoaderConfig()
    optimizer: torch_core_config.OptimizerConfig = torch_core_config.TorchOptimizerConfig()
    number_of_epochs: int = 2
    finetune_only_head: bool = False
    target_attribute: str = "default_"


config_store = hydra_config_store.ConfigStore.instance()
config_store.store(group="adaptation", name="no_adaptation", node=NoAdaptationConfig)
config_store.store(group="adaptation", name="finetune", node=FineTuneConfig)
