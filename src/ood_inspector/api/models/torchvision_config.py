import dataclasses

import omegaconf

from ood_inspector.api.models import base_config
from ood_inspector.datasets import IMAGENET_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from ood_inspector.models import torchvision as torchvision_models


@dataclasses.dataclass
class TorchvisionModelConfig(base_config.ModelConfig):
    _target_: str = "ood_inspector.models.torchvision.load_torchvision_model"
    name: torchvision_models.TorchvisionModelName = omegaconf.MISSING


def register_configs(pretrained: bool = True) -> None:
    def config_name(name: str) -> str:
        return base_config.config_name(name, "torchvision", pretrained)

    def config_node(name: str) -> base_config.ModelConfig:
        if pretrained:
            pretraining_config = {
                "pretraining_input_size": IMAGENET_INPUT_SIZE,
                "pretraining_input_mean": IMAGENET_MEAN,
                "pretraining_input_std": IMAGENET_STD,
            }
        else:
            pretraining_config = {}
        return TorchvisionModelConfig(name=name, pretrained=pretrained, **pretraining_config)

    base_config.register_list_of_models(
        torchvision_models.list_torchvision_models(pretrained=pretrained), config_name, config_node
    )
