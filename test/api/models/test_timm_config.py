from typing import Callable, Optional, Tuple

import hydra.utils
import pytest

try:
    from ood_inspector.api.models import timm_config
    from ood_inspector.models import timm as timm_models
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


@pytest.mark.parametrize("pretrained", [True, False])
def test_timm_model_config_with_string(pretrained):
    assert timm_config.TimmModelConfig("resnet18", pretrained)


def test_timm_model_config_with_pretrained_timm_model():
    assert timm_config.TimmModelConfig(timm_models.TimmModelName.resnet18)


@pytest.mark.parametrize(
    "model_name",
    [
        "densenet121",
        "resnet18",
        "resnet152d",
    ],
)
@pytest.mark.parametrize("pretrained", [True, False])
def test_timm_models_register_successfully(
    inspector_config: Callable,
    model_name: str,
    pretrained: bool,
) -> None:
    """Test timm model configuration with different model names."""
    timm_config.register_configs(pretrained=pretrained)
    argument = f"+model=timm_{model_name}"
    if pretrained:
        argument = f"+model=timm_pretrained_{model_name}"

    cfg = inspector_config([argument])
    assert cfg.model.name == timm_models.TimmModelName[model_name]


@pytest.mark.parametrize(
    "argument",
    [
        "+model=timm_densenet121",
        "+model=timm_resnet18",
    ],
)
def test_timm_models_instantiate_correct_model(inspector_config: Callable, argument: str) -> None:
    timm_config.register_configs(pretrained=False)
    cfg = inspector_config([argument])
    model = hydra.utils.instantiate(cfg.model)
    assert isinstance(model, timm_models.TimmModel)


@pytest.mark.parametrize(
    "model_name,pretraining_input_size,pretraining_input_mean,pretraining_input_std",
    [
        ("timm_resnet18", None, None, None),
        (
            "timm_pretrained_resnet18",
            (3, 224, 224),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        (
            "timm_pretrained_tf_efficientnet_b5",
            (3, 456, 456),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        (
            "timm_pretrained_vit_base_patch16_224",
            (3, 224, 224),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ],
)
def test_timm_input_config(
    inspector_config: Callable,
    model_name: str,
    pretraining_input_size: Optional[Tuple[int, int, int]],
    pretraining_input_mean: Optional[Tuple[float, float, float]],
    pretraining_input_std: Optional[Tuple[float, float, float]],
) -> None:
    for pretrained in [True, False]:
        timm_config.register_configs(pretrained=pretrained)

    cfg = inspector_config([f"+model={model_name}"])
    model = hydra.utils.instantiate(cfg.model)

    assert model.pretraining_input_size == pretraining_input_size
    assert model.pretraining_input_mean == pretraining_input_mean
    assert model.pretraining_input_std == pretraining_input_std
