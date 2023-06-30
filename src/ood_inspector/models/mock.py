"""Inspector models"""
import dataclasses
from typing import Optional, Tuple

import torch

from ood_inspector.models import inspector_base


@dataclasses.dataclass
class MockModel(inspector_base.InspectorModel):
    """Mock model for testing and documentation."""

    name: str
    pretrained: bool
    _setup: bool = False

    def __init__(
        self,
        device,
        pretraining_input_size: Optional[Tuple[int, int, int]] = None,
        pretraining_input_mean: Optional[Tuple[float, float, float]] = None,
        pretraining_input_std: Optional[Tuple[float, float, float]] = None,
    ):
        self._device = device
        # Pass the simplest `nn.Module` possible to parent class constructor. This ensures the class
        # is correctly initialized.
        super().__init__(
            torch.nn.Identity(),
            pretraining_input_size,
            pretraining_input_mean,
            pretraining_input_std,
        )

    @property
    def is_normalized(self) -> bool:
        return False

    @property
    def n_classes(self) -> int:
        return None

    @property
    def n_features(self) -> int:
        return None

    @property
    def number_of_parameters(self) -> int:
        return 0

    def forward(self, *args, **kwargs) -> inspector_base.InspectorModelOutput:
        return inspector_base.InspectorModelOutput(torch.Tensor(1), torch.Tensor(1))

    def setup(self, _input_batch: torch.Tensor) -> None:
        self._setup = True

    @property
    def device(self):
        """Need to override the parent function as we neither have buffers nor parameters."""
        return self._device

    def set_classification_head(self, number_of_classes: int):
        pass
