import abc
import dataclasses
import logging
from typing import Any

import torch

from ood_inspector import torch_core
from ood_inspector.datasets.dataset import InspectorDataset
from ood_inspector.models import inspector_base


class ModelAdaptation(metaclass=abc.ABCMeta):
    """Adaptation class to update the weights of a model."""

    @abc.abstractmethod
    def fit(self, model: inspector_base.InspectorModel) -> inspector_base.InspectorModel:
        pass


@dataclasses.dataclass
class NoAdaptation(ModelAdaptation):
    def fit(self, model: inspector_base.InspectorModel) -> inspector_base.InspectorModel:
        return model


class FineTune(ModelAdaptation):
    """Finetuning class with VTAB defaults."""

    def __init__(
        self,
        dataset: InspectorDataset,
        dataloader: Any,
        optimizer: Any,
        lr_scheduler: Any,
        number_of_epochs: int,
        finetune_only_head: bool = False,
        target_attribute: str = "default_"
        # TODO(pgehler): Add criterion as an option.
    ):
        self.dataset = dataset
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.number_of_epochs = number_of_epochs
        self.finetune_only_head = finetune_only_head
        if target_attribute == "default_":
            target_attribute = dataset.default_attribute
        self.target_attribute = target_attribute

    def fit(self, model: inspector_base.InspectorModel) -> inspector_base.InspectorModel:

        if self.finetune_only_head:
            logging.info("Finetuning last layer only")
            for parameter in model.parameters():
                parameter.requires_grad = False
        else:
            logging.info("Finetuning all network parameters")

        model.set_classification_head(
            self.dataset.number_of_classes_per_attribute[self.target_attribute]
        )

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                logging.debug(f"Layer {name} will be updated.")

        torch_optimizer = self.optimizer(model.parameters())

        lr_scheduler = self.lr_scheduler(torch_optimizer) if self.lr_scheduler else None
        if lr_scheduler:
            logging.info(f"Using learning rate scheduler: {lr_scheduler.__class__.__name__}")
        else:
            logging.info("No learning rate scheduler used.")

        self.model = torch_core.fit(
            model=model,
            device=model.device,
            data_source=self.dataloader(self.dataset),
            criterion=torch.nn.CrossEntropyLoss().to(model.device),
            optimizer=torch_optimizer,
            lr_scheduler=lr_scheduler,
            number_of_epochs=self.number_of_epochs,
            target_attribute=self.target_attribute,
        )
        # Since torch_core.fit updates model in-place, model = self.model.
        return model
