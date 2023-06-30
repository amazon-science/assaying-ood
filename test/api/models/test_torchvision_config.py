from typing import Callable, Optional, Tuple

import hydra.utils
import pytest

# TODO(pgehler): skip all tests if there are modules not available in the versionset.
try:
    from ood_inspector.api.models import torchvision_config
    from ood_inspector.models import torchvision
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


@pytest.mark.parametrize("pretrained", [True, False])
def test_torchvisionmodel_config_with_string(pretrained):
    assert torchvision_config.TorchvisionModelConfig("resnet18", pretrained)


def test_torchvision_model_config_with_pretrained_model():
    assert torchvision_config.TorchvisionModelConfig(torchvision.TorchvisionModelName.resnet18)


@pytest.mark.parametrize(
    "model_name",
    [
        "densenet121",
        "vgg16",
        "mobilenet_v2",
    ],
)
@pytest.mark.parametrize("pretrained", [True, False])
def test_torchvision_models_register_successfully(
    inspector_config: Callable,
    model_name: str,
    pretrained: bool,
) -> None:
    """Test torchvision model configuration with different model names."""

    torchvision_config.register_configs(pretrained=pretrained)
    argument = f"+model=torchvision_{model_name}"
    if pretrained:
        argument = f"+model=torchvision_pretrained_{model_name}"

    cfg = inspector_config([argument])
    assert cfg.model.name == torchvision.TorchvisionModelName[model_name]


@pytest.mark.parametrize(
    "argument",
    [
        "+model=torchvision_densenet121",
        "+model=torchvision_vgg16",
        "+model=torchvision_mobilenet_v2",
    ],
)
def test_torchvision_models_instantiate_correct_model(
    inspector_config: Callable, argument: str
) -> None:
    torchvision_config.register_configs(pretrained=False)

    cfg = inspector_config([argument])
    model = hydra.utils.instantiate(cfg.model)
    assert isinstance(model, torchvision.TorchvisionModel)


def test_only_torchvision_models_registered(inspector_config):
    with pytest.raises(hydra.errors.MissingConfigException):
        inspector_config(["+model=torchvision_does_not_exist"])


@pytest.mark.parametrize(
    "model_name,pretraining_input_size,pretraining_input_mean,pretraining_input_std",
    [
        ("torchvision_resnet18", None, None, None),
        (
            "torchvision_pretrained_resnet18",
            (3, 224, 224),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        ("torchvision_mobilenet_v2", None, None, None),
        (
            "torchvision_pretrained_mobilenet_v2",
            (3, 224, 224),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ],
)
def test_torchvision_input_config(
    inspector_config: Callable,
    model_name: str,
    pretraining_input_size: Optional[Tuple[int, int, int]],
    pretraining_input_mean: Optional[Tuple[float, float, float]],
    pretraining_input_std: Optional[Tuple[float, float, float]],
) -> None:
    for pretrained in [True, False]:
        torchvision_config.register_configs(pretrained=pretrained)

    cfg = inspector_config([f"+model={model_name}"])
    model = hydra.utils.instantiate(cfg.model)

    assert model.pretraining_input_size == pretraining_input_size
    assert model.pretraining_input_mean == pretraining_input_mean
    assert model.pretraining_input_std == pretraining_input_std
