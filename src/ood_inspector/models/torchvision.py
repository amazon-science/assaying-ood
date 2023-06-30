"""Inspector models"""
import enum
import inspect
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

from ood_inspector.models import inspector_base


def list_torchvision_models(pretrained: bool = True) -> List[str]:
    """Get list of all available pretrained models."""
    valid_models = []
    for model_name, model_object in vars(torchvision.models).items():
        if not isinstance(model_object, Callable):
            continue
        parameters = inspect.signature(model_object).parameters
        has_pretrained_argument = any(parameter == "pretrained" for parameter in parameters)
        if has_pretrained_argument:
            valid_models.append(model_name)
    return valid_models


TorchvisionModelName = enum.Enum("TorchvisionModelName", list_torchvision_models(pretrained=True))


class TorchvisionModel(inspector_base.InspectorModel):
    def __init__(
        self,
        model: nn.Module,
        pretraining_input_size: Optional[Tuple[int, int, int]] = None,
        pretraining_input_mean: Optional[Tuple[float, float, float]] = None,
        pretraining_input_std: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__(
            model, pretraining_input_size, pretraining_input_mean, pretraining_input_std
        )
        # TODO(cjsg): Set n_classes and n_features for torchvision models.
        # (Currently, this is being set only when using set_classification_net, which currently
        # works only for ResNets and VGG)
        self._n_classes = None
        self._n_features = None

    @property
    def is_normalized(self) -> bool:
        return False

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def n_features(self) -> int:
        return self._n_features

    def setup(self) -> None:
        pass

    def forward(self, inputs: torch.Tensor) -> inspector_base.InspectorModelOutput:
        logits = self.model(inputs)
        return inspector_base.InspectorModelOutput(logits=logits, features=None)

    @property
    def _model_name(self) -> str:
        return self.model.__class__.__name__

    def set_classification_head(self, number_of_classes: int) -> None:
        if self._model_name == "ResNet":
            self._n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, number_of_classes)
        elif self._model_name == "VGG":
            self._n_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(self.n_features, number_of_classes)
        else:
            raise NotImplementedError(f"{self._model_name} not supported")
        self._n_classes = number_of_classes
        self.model.to(self.device)


def load_torchvision_model(
    name: enum.Enum,
    pretrained: bool,
    device: str,
    pretraining_input_size: Optional[Tuple[int, int, int]] = None,
    pretraining_input_mean: Optional[Tuple[float, float, float]] = None,
    pretraining_input_std: Optional[Tuple[float, float, float]] = None,
    **kwargs,
) -> TorchvisionModel:
    """Wrapper function to create a torchvision model."""
    model_cls = getattr(torchvision.models, name.name)
    return TorchvisionModel(
        model_cls(pretrained=pretrained, **kwargs).to(torch.device(device)),
        pretraining_input_size,
        pretraining_input_mean,
        pretraining_input_std,
    )
