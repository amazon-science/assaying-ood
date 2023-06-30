import dataclasses

import omegaconf
import timm

from ood_inspector.api.models import base_config
from ood_inspector.models import timm as timm_models


@dataclasses.dataclass
class TimmModelConfig(base_config.ModelConfig):
    _target_: str = "ood_inspector.models.timm.load_timm_model"
    name: timm_models.TimmModelName = omegaconf.MISSING


def register_configs(pretrained: bool = True) -> None:
    def config_name(name: str) -> str:
        return base_config.config_name(name, "timm", pretrained)

    def config_node(name: str) -> base_config.ModelConfig:
        if pretrained:
            config = timm.models.registry._model_default_cfgs[name]
            pretraining_config = {
                "pretraining_input_size": config["input_size"],
                "pretraining_input_mean": config["mean"],
                "pretraining_input_std": config["std"],
            }
        else:
            pretraining_config = {}
        return TimmModelConfig(name=name, pretrained=pretrained, **pretraining_config)

    base_config.register_list_of_models(
        timm.list_models(pretrained=pretrained), config_name, config_node
    )
