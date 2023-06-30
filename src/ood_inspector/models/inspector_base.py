"""Inspector models"""
import abc
import dataclasses
import functools
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ood_inspector import utils


def _ensure_setup(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self._setup:
            raise RuntimeError("model.setup() must be called before any other method.")
        return method(self, *args, **kwargs)

    return wrapped


@dataclasses.dataclass
class InspectorModelOutput:
    """Output of an Inspector model."""

    # TODO(hornmax): It looks like we do not respect this anymore and that instead the model has
    # a `is_normalized` property. We should thus rename this here to ensure it does not lead to
    # confusion.
    logits: torch.Tensor
    features: Optional[torch.Tensor] = None

    def to(self, device):
        return self.__class__(
            *(getattr(self, field.name).to(device) for field in dataclasses.fields(self))
        )

    def __getitem__(self, idx):
        features_batch = None if self.features is None else self.features[idx]
        return InspectorModelOutput(self.logits[idx], features_batch)


class InspectorModel(nn.Module, metaclass=abc.ABCMeta):
    """Common class for inspector models.

    Further, gives access to important properties of the model, such as the number of classes and
    the number of features in the penultimate layer.
    """

    def __init__(
        self,
        model: nn.Module,
        pretraining_input_size: Optional[Tuple[int, int, int]] = None,
        pretraining_input_mean: Optional[Tuple[float, float, float]] = None,
        pretraining_input_std: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Args:
            model (nn.Module): Model.
        """

        super().__init__()
        self.model = model
        self._pretraining_input_size = pretraining_input_size
        self._pretraining_input_mean = pretraining_input_mean
        self._pretraining_input_std = pretraining_input_std
        self._setup = False

    @abc.abstractmethod
    def setup(self, input_batch: torch.Tensor) -> None:
        self._setup = True

    @abc.abstractmethod
    @_ensure_setup
    def forward(self, *args, **kwargs) -> InspectorModelOutput:
        pass

    @abc.abstractmethod
    @_ensure_setup
    def set_classification_head(self, number_of_classes: int) -> None:
        # TODO(cjsg): Create a FintuneableModel class where this method is required.
        # Because for evaluations only, you don't need this method.
        pass

    @abc.abstractproperty
    @_ensure_setup
    def n_classes(self) -> int:
        pass

    @abc.abstractproperty
    @_ensure_setup
    def n_features(self) -> int:
        pass

    @abc.abstractproperty
    def is_normalized(self) -> bool:
        """If true, it indicates that the model output is normalized (i.e a probability vector)."""
        pass

    @property
    def device(self) -> str:
        return utils.get_device_from_module(self.model)

    @property
    def number_of_parameters(self) -> int:
        return sum([parameter.numel() for parameter in self.parameters()])

    @property
    def pretraining_input_size(self) -> Optional[Tuple[int, int, int]]:
        return self._pretraining_input_size

    @property
    def pretraining_input_mean(self) -> Optional[Tuple[float, float, float]]:
        return self._pretraining_input_mean

    @property
    def pretraining_input_std(self) -> Optional[Tuple[float, float, float]]:
        return self._pretraining_input_std


class ModelWrapperForLogits(nn.Module):
    """Utility class to make an inspector model return logits and add a pre-processing layer."""

    def __init__(
        self, model: InspectorModel, mean: Sequence[float] = (0.0,), std: Sequence[float] = (1.0,)
    ):
        super().__init__()
        self.model = model
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean.view(-1, 1, 1))
        self.register_buffer("std", std.view(-1, 1, 1))
        self.train(mode=model.training)
        self.to(model.device)

    def forward(self, inputs):
        inputs = inputs - self.mean
        inputs = inputs / self.std
        return self.model(inputs).logits
