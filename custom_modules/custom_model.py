import torch
import dataclasses

import hydra.core.config_store as hydra_config_store

from ood_inspector.models.inspector_base import InspectorModel, InspectorModelOutput
from ood_inspector.api.models.base_config import ModelConfig

from typing import Tuple


class FCNet(torch.nn.Module):
    def __init__(self, input_size, number_of_features, number_of_classes=2):
        super().__init__()
        self.input_size = input_size
        self.number_of_features = number_of_features
        self.number_of_classes = number_of_classes

        w, h, c = input_size
        self.fc1 = torch.nn.Linear(in_features=w * h * c, out_features=number_of_features)
        self.fc2 = torch.nn.Linear(in_features=number_of_features, out_features=number_of_features)
        self.classifier = torch.nn.Linear(
            in_features=number_of_features, out_features=number_of_classes
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x


class FCNetInspectorModel(InspectorModel):
    """Wrapper for making the FCNet model compatible with the Inspector pipeline."""

    def __init__(self, model: torch.nn.Module, device: str, **kwargs):
        model.to(device)
        super().__init__(model)

    def forward(self, x):
        logits = self.model(x)
        return InspectorModelOutput(logits=logits, features=None)

    def set_classification_head(self, number_of_classes: int) -> None:
        self.model.classifier = torch.nn.Linear(
            in_features=self.n_features, out_features=number_of_classes
        )
        self.model.to(self.device)
        self.model.number_of_classes = number_of_classes

    def setup(self, input_batch: torch.Tensor) -> None:
        super().setup(input_batch)

    @property
    def n_classes(self) -> int:
        return self.model.number_of_classes

    @property
    def n_features(self) -> int:
        return self.model.number_of_features

    @property
    def is_normalized(self) -> bool:
        return False


@dataclasses.dataclass
class FCNetConfig:
    _target_: str = "custom_modules.custom_model.FCNet"
    input_size: Tuple[int] = (3, 224, 224)
    number_of_features: int = 64
    number_of_classes: int = 2


@dataclasses.dataclass
class FCNetInspectorModelConfig(ModelConfig):
    _target_: str = "custom_modules.custom_model.FCNetInspectorModel"
    model: FCNetConfig = dataclasses.field(default_factory=lambda: FCNetConfig())
    device: str = "cuda"


print("Registering custom model!")
config_store = hydra_config_store.ConfigStore.instance()
config_store.store(group="model", name="fcnet", node=FCNetInspectorModelConfig)
