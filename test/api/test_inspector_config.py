import hydra
import pytest

try:
    from ood_inspector.api import inspector_config  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


# TODO(pgehler): Implement hydra testing. https://hydra.cc/docs/advanced/unit_testing
def test_configs_registered() -> None:
    """Test creation of config."""
    with hydra.initialize_config_module(version_base="1.1", config_module="ood_inspector"):
        cfg = hydra.compose(
            config_name="inspector",
            overrides=["+model=mock", "s3_output_path=s3://bucket/some/folder"],
        )
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "s3_output_path")
        assert cfg["s3_output_path"] == "s3://bucket/some/folder"


def test_load_yaml_configs() -> None:
    """Test creation of config based on Yaml."""
    with hydra.initialize(version_base="1.1", config_path="_test-conf"):
        cfg = hydra.compose(config_name="test_inspector")
        assert (
            cfg["datasets"]["dataset1"]["dataset"]["_target_"]
            == "ood_inspector.datasets.webdataset.get_webdataset"
        )
