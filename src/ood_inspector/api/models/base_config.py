import dataclasses
from typing import Callable, List, Optional, Tuple

import hydra.core.config_store as hydra_config_store
import torch


@dataclasses.dataclass
class ModelConfig:
    """Model configuration.

    If ``pretrained`` is ``True`` the pretraining input configuration variables
    ``pretraining_input_size``, ``pretraining_input_mean``, ``pretraining_input_std`` should also be
    set. Otherwise, they will default to ``None``.

    Args:
        device: Target device where to load the model.
        pretrained: Whether the loaded model was pretrained or not.
        pretraining_input_size: Input size of the images during pretraining.
        pretraining_input_mean: Mean used for normalization during pretraining.
        pretraining_input_std: Standard deviation used for normaization during pretraining.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained: bool = False
    pretraining_input_size: Optional[Tuple[int, int, int]] = None
    pretraining_input_mean: Optional[Tuple[float, float, float]] = None
    pretraining_input_std: Optional[Tuple[float, float, float]] = None


def config_name(model_name: str, prefix: str, pretrained: bool = True):
    if pretrained:
        return f"{prefix}_pretrained_{model_name}"
    return f"{prefix}_{model_name}"


def register_list_of_models(
    model_names: List[str],
    config_name: Callable[[str], str],
    config_node: Callable[[str], ModelConfig],
) -> None:

    config_store = hydra_config_store.ConfigStore.instance()
    for model_name in model_names:
        config_store.store(
            group="model",
            name=config_name(model_name),
            node=config_node(model_name),
        )
