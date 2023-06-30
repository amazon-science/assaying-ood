from typing import Callable, Tuple

import hydra
import pytest

try:
    from ood_inspector.api.datasets import datasets_config  # noqa: F401
    from ood_inspector.api.models import timm_config, torchvision_config
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


# TODO(pgehler): Add all datasets programatically.
@pytest.mark.parametrize("dataset_name", ["S3ImageNet1k-val"])
def test_datasets_register_successfully(inspector_config: Callable, dataset_name: str) -> None:
    """Test datasets configuration with different dataset names."""
    cfg = inspector_config([f"+datasets={dataset_name}"])
    assert hasattr(cfg, "datasets")
    assert hasattr(cfg.datasets, dataset_name)
    assert hasattr(cfg.datasets[dataset_name], "_target_")


def test_only_valid_datasets_registered(inspector_config):
    with pytest.raises(hydra.errors.MissingConfigException):
        inspector_config(["+datasets=does_not_exist"])


@pytest.mark.parametrize(
    "model_name,input_size,input_mean,input_std",
    [
        ("torchvision_resnet18", (3, 224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        (
            "torchvision_pretrained_resnet18",
            (3, 224, 224),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        ("timm_vit_base_patch16_224", (3, 224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        (
            "timm_pretrained_vit_base_patch16_224",
            (3, 224, 224),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ],
)
def test_dataset_transformation_config(
    inspector_config: Callable,
    model_name: str,
    input_size: Tuple[int, int, int],
    input_mean: Tuple[float, float, float],
    input_std: Tuple[float, float, float],
) -> None:
    for pretrained in [True, False]:
        timm_config.register_configs(pretrained=pretrained)
        torchvision_config.register_configs(pretrained=pretrained)

    cfg = inspector_config([f"+model={model_name}", "+datasets=S3DomainBed-PACS-cartoon"])
    dataset_cfg = cfg.datasets["S3DomainBed-PACS-cartoon"]

    assert dataset_cfg.transformations.transformation.input_size == input_size
    assert dataset_cfg.transformations.transformation.input_mean == input_mean
    assert dataset_cfg.transformations.transformation.input_std == input_std
