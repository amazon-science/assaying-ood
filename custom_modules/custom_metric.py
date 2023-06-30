import torch
import torch.nn as nn
import dataclasses

import hydra.core.config_store as hydra_config_store

from ood_inspector.evaluations import Evaluation
from ood_inspector.models.inspector_base import InspectorModel
from ood_inspector.api.evaluations_config import EvaluationConfig

from typing import Any, Dict


class CustomTop1ClassificationAccuracy(nn.Module, Evaluation):

    n_correct: torch.Tensor
    n_scored: torch.Tensor

    def __init__(self, target_attribute: str = "default_") -> None:
        """
        Args:
            target_attribute: Attribute with respect to which we evaluate.
        """
        nn.Module.__init__(self)
        Evaluation.__init__(self, target_attribute)

    def setup(
        self,
        model: InspectorModel,
        normalization_transform: Any,
    ):
        try:
            self.n_correct.zero_()
            self.n_scored.zero_()
        except AttributeError:
            self.register_buffer("n_correct", torch.zeros((1,), dtype=torch.int64, device="cpu"))
            self.register_buffer("n_scored", torch.zeros((1,), dtype=torch.int64, device="cpu"))

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        del inputs
        labels = all_labels[self.target_attribute]
        predicted_class = outputs.argmax(-1)
        self.n_correct.add_((predicted_class == labels).long().sum())
        self.n_scored.add_(labels.numel())

    def score(self) -> float:
        return float(self.n_correct / self.n_scored)

    def __str__(self) -> str:
        return "Top-1 classification accuracy"

    @property
    def requires_data(self) -> bool:
        return True


@dataclasses.dataclass
class CustomMetricConfig(EvaluationConfig):
    _target_: str = "custom_modules.custom_metric.CustomTop1ClassificationAccuracy"


print("Registering custom metric!")
config_store = hydra_config_store.ConfigStore.instance()
config_store.store(
    group="evaluations", name="custom_metric", node={"custom_metric": CustomMetricConfig}
)
