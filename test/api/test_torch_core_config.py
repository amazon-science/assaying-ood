import hydra
import pytest

try:
    from ood_inspector.api import inspector_config, torch_core_config  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


# TODO(pgehler): Add all datasets programatically.
@pytest.mark.parametrize("dataloader_name", ["TorchDataLoader"])
def test_dataloaders_register_successfully(dataloader_name: str):
    """Test datasets configuration with different dataset names."""
    with hydra.initialize_config_module(version_base="1.1", config_module="ood_inspector"):
        cfg = hydra.compose(
            config_name="inspector",
            overrides=[f"dataloader={dataloader_name}"],
        )
        assert hasattr(cfg, "dataloader")


def test_dataloaders_options_can_be_set():
    """Test datasets configuration with different dataset names."""
    with hydra.initialize_config_module(version_base="1.1", config_module="ood_inspector"):
        cfg = hydra.compose(
            config_name="inspector",
            overrides=["dataloader.batch_size=2"],
        )
        assert hasattr(cfg, "dataloader")
        assert hasattr(cfg.dataloader, "batch_size")
        assert cfg.dataloader.batch_size == 2


def test_vtab_lr_scheduler_config():
    """Test multistep_30_60_90 scheduler config correctly interpolates number_of_epochs."""
    with hydra.initialize_config_module(version_base="1.1", config_module="ood_inspector"):
        cfg = hydra.compose(
            config_name="inspector",
            overrides=[
                "adaptation=finetune",
                "adaptation.number_of_epochs=10",
                "adaptation/lr_scheduler=multistep_30_60_90",
            ],
        )
        assert cfg.adaptation.lr_scheduler.classname == "MultiStepLR"
        assert cfg.adaptation.lr_scheduler.options.gamma == 0.1
        assert cfg.adaptation.lr_scheduler.options.milestones == [3, 6, 9]
