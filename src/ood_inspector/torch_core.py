import dataclasses
import logging
import sys
import time
from typing import Any, ClassVar, Dict, List, Optional

import torch.nn
import torch.optim
import torch.utils.data
import tqdm

from ood_inspector.models import inspector_base


@dataclasses.dataclass
class DataLoader:
    pass


@dataclasses.dataclass
class TorchDataLoader(DataLoader):
    """PyTorch DataLoader."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataset) -> torch.utils.data.DataLoader:
        if dataset is None:
            return None
        return torch.utils.data.DataLoader(dataset, *self.args, **self.kwargs)


@dataclasses.dataclass
class TorchOptimizer:
    classname: str
    defaults: Optional[Dict]

    def __call__(self, parameters) -> torch.optim.Optimizer:
        """Returns torch optimizer.

        This implements a functor that returns a Torch optimizer, so to allow
        to configure it separately. The following expressions are equivalent::

            optimizer = torch.optim.SGD(parameters, lr=0.1)

            opt = TorchOptimizer('SGD', {'lr': 0.1})
            optimizer = opt(parameters)
        """
        cls = getattr(sys.modules["torch.optim"], self.classname)
        return cls(parameters, **self.defaults)


@dataclasses.dataclass
class LRScheduler:
    pass


@dataclasses.dataclass
class TorchLRScheduler(LRScheduler):
    classname: str
    options: Optional[Dict]

    SUPPORTED_TYPES: ClassVar[List] = [
        torch.optim.lr_scheduler.CosineAnnealingLR,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        torch.optim.lr_scheduler.ExponentialLR,
        torch.optim.lr_scheduler.MultiStepLR,
        torch.optim.lr_scheduler.StepLR,
    ]

    def __call__(self, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Returns torch lr scheduler.

        This implements a functor that returns a Torch learning rate scheduler, so to allow to
        configure it separately. The following expressions are equivalent::

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=[10, 20, 30], gamma=0.1
            )

            scheduler = TorchLRScheduler('MultiStepLR', {'milestones': [10, 20, 30], 'gamma': 0.1})
            lr_scheduler = scheduler(optimizer)
        """
        cls = getattr(sys.modules["torch.optim.lr_scheduler"], self.classname)

        if cls not in TorchLRScheduler.SUPPORTED_TYPES:
            raise ValueError(f"{self.classname} scheduler is not supported.")

        return cls(optimizer, **self.options)


def fit(
    model: torch.nn.Module,
    device: Any,
    data_source: Any,
    criterion,
    optimizer: torch.optim.Optimizer,
    number_of_epochs: int,
    target_attribute: str,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> torch.nn.Module:
    """Model training

    Trains a given model with the provided dataset, criterion and optimizer.
    """
    start_time = time.time()
    if isinstance(model, inspector_base.InspectorModel):
        model = inspector_base.ModelWrapperForLogits(model)

    model.train()
    iterator_with_progress = tqdm.tqdm(data_source, total=len(data_source), desc="Model training")
    for epoch in range(number_of_epochs):
        logging.info(f"Epoch {epoch}/{number_of_epochs-1}")
        running_loss = 0.0
        for sample in iterator_with_progress:
            images = sample["image"].to(device)
            labels = sample[target_attribute].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(data_source)
        learning_rate = [param_group["lr"] for param_group in optimizer.param_groups]
        logging.info(f" Learning Rate {learning_rate[0]}")
        logging.info(f" Loss {epoch_loss}")

        if lr_scheduler:
            lr_scheduler.step()

    total_training_time = time.time() - start_time
    logging.info(f"Training complete in {total_training_time // 60}m {total_training_time % 60}s")

    if isinstance(model, inspector_base.ModelWrapperForLogits):
        model = model.model
    return model
