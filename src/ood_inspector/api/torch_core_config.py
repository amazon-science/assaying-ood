import dataclasses
from typing import Dict, List

import hydra.core.config_store as hydra_config_store
import omegaconf


@dataclasses.dataclass
class DataLoaderConfig:
    pass


@dataclasses.dataclass
class TorchDataLoaderConfig(DataLoaderConfig):
    _target_: str = "ood_inspector.torch_core.TorchDataLoader"
    batch_size: int = 32
    num_workers: int = 0


@dataclasses.dataclass
class OptimizerConfig:
    pass


@dataclasses.dataclass
class TorchOptimizerConfig(OptimizerConfig):
    _target_: str = "ood_inspector.torch_core.TorchOptimizer"
    classname: str = "SGD"
    defaults: Dict = dataclasses.field(default_factory=lambda: {})


@dataclasses.dataclass
class LRSchedulerConfig:
    pass


@dataclasses.dataclass
class TorchLRSchedulerConfig(LRSchedulerConfig):
    _target_: str = "ood_inspector.torch_core.TorchLRScheduler"
    classname: str = omegaconf.MISSING
    options: Dict = dataclasses.field(default_factory=lambda: {})


config_store = hydra_config_store.ConfigStore.instance()
config_store.store(group="dataloader", name="TorchDataLoader", node=TorchDataLoaderConfig)
config_store.store(group="adaptation/lr_scheduler", name="torch", node=TorchLRSchedulerConfig)


# Registering the lr scheduler used in the VTAB paper (https://arxiv.org/abs/1910.04867).
# github.com/google-research/task_adaptation/blob/master/scripts/run_all_tasks.sh
# MultiStepLR with 3 milestones after 30%, 60% and 90% of the total number of epochs.
def convert_fractional_to_integer_steps(steps: List[float], number_of_epochs: int) -> List[int]:
    return [int(step * number_of_epochs) for step in steps]


omegaconf.OmegaConf.register_new_resolver(
    "convert_fractional_to_integer_steps", convert_fractional_to_integer_steps
)


config_store.store(
    group="adaptation/lr_scheduler",
    name="multistep_30_60_90",
    node=TorchLRSchedulerConfig(
        classname="MultiStepLR",
        options={
            "milestones": omegaconf.SI(
                (
                    "${convert_fractional_to_integer_steps:[0.3,0.6,0.9],"
                    "${adaptation.number_of_epochs}}"
                )
            ),
            "gamma": 0.1,
        },
    ),
)
