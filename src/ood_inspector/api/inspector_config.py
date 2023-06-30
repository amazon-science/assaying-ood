"""Base configuration for inspector runs."""
import dataclasses
from typing import Any, Dict, List, Optional

import hydra.core.config_store as hydra_config_store
import omegaconf

from ood_inspector.api import (
    adaptation_config,
    corruption_config,
    evaluations_config,
    torch_core_config,
)
from ood_inspector.api.datasets import datasets_config
from ood_inspector.api.models import base_config as models_config

_INSPECTOR_DEFAULTS = [
    {"adaptation": "no_adaptation"},
    {"corruption": "no_corruption"},
    {"dataloader": "TorchDataLoader"},
    {"evaluations": []},
    "_self_",
]


@dataclasses.dataclass
class InspectorConfig:
    """Main configuration of a Inspector run."""

    _target_: str = "ood_inspector.inspector.Inspector"
    defaults: List[Any] = dataclasses.field(default_factory=lambda: _INSPECTOR_DEFAULTS)

    s3_output_path: Optional[str] = None
    model: models_config.ModelConfig = omegaconf.MISSING
    corruption: corruption_config.CorruptionConfig = omegaconf.MISSING
    datasets: Dict[str, datasets_config.InspectorDatasetConfig] = dataclasses.field(
        default_factory=dict
    )
    dataloader: torch_core_config.DataLoaderConfig = omegaconf.MISSING
    evaluations: Dict[str, evaluations_config.EvaluationConfig] = omegaconf.MISSING
    adaptation: adaptation_config.AdaptationConfig = omegaconf.MISSING
    save_inspector: bool = False


config_store = hydra_config_store.ConfigStore.instance()
config_store.store(name="inspector", node=InspectorConfig)
