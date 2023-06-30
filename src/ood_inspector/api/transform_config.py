import dataclasses
from typing import Tuple

import hydra.core.config_store as hydra_config_store
import omegaconf


def use_default_if_none(config, default_value):
    if config is None:
        return tuple(default_value)
    return config


omegaconf.OmegaConf.register_new_resolver("use_default_if_none", use_default_if_none)


@dataclasses.dataclass
class TransformConfig:
    """Image transformation configuration useful for basic processing as resizing and normalization.

    If no values are set, the default configuration is inherited from the input transformation
    configuration the model used for pretraining. If the model doesn't have a pretraining
    configuration, the default value for ``input_size`` will be (3, 224, 224) and the ``input_mean``
    and ``input_std`` will be set to ``(0.0, 0.0, 0.0)`` and ``(1.0, 1.0, 1.0)``, respectively,
    corresponding to an identity normalization transformation.
    """

    input_size: Tuple[int, int, int] = omegaconf.SI(
        "${use_default_if_none:${model.pretraining_input_size},[3,224,224]}"
    )
    input_mean: Tuple[float, float, float] = omegaconf.SI(
        "${use_default_if_none:${model.pretraining_input_mean},[0.0,0.0,0.0]}"
    )
    input_std: Tuple[float, float, float] = omegaconf.SI(
        "${use_default_if_none:${model.pretraining_input_std},[1.0,1.0,1.0]}"
    )


@dataclasses.dataclass
class DefaultTransformConfig(TransformConfig):
    _target_: str = "ood_inspector.transform.load_base_transform"
    training: bool = omegaconf.MISSING


@dataclasses.dataclass
class AdaptationTransformConfig(DefaultTransformConfig):
    training: bool = True


@dataclasses.dataclass
class EvaluationTransformConfig(DefaultTransformConfig):
    training: bool = False


config_store = hydra_config_store.ConfigStore.instance()
config_store.store(group="schemas/transforms", name="DefaultTransform", node=DefaultTransformConfig)
config_store.store(
    group="schemas/transforms",
    name="AdaptationTransform",
    node=AdaptationTransformConfig,
)
config_store.store(
    group="schemas/transforms",
    name="EvaluationTransform",
    node=EvaluationTransformConfig,
)
