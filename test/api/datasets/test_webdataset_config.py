import hydra
import pytest

try:
    from ood_inspector.api.datasets import webdataset_config
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_webdataset_config() -> None:
    assert webdataset_config.WebDatasetConfig(
        uri_expression="does_not_exist.tar",
        number_of_datapoints=1,
        number_of_classes_per_attribute={"": 100},
    )


def test_s3datasets_register() -> None:
    """Test creation of config based on Yaml."""
    with hydra.initialize(version_base="1.1", config_path="_test-conf"):
        cfg = hydra.compose(config_name="test_s3_datasets")
        assert (
            cfg["datasets"]["dataset1"]["dataset"]["_target_"]
            == "ood_inspector.datasets.webdataset.get_webdataset"
        )
        assert (
            cfg["datasets"]["dataset2"]["dataset"]["dataset"]["_target_"]
            == "ood_inspector.datasets.webdataset.get_webdataset"
        )
        assert cfg.datasets["dataset1"]["dataset"]["number_of_datapoints"] == 50000
        assert (
            cfg.datasets["dataset1"]["dataset"]["number_of_classes_per_attribute"]["label_"] == 1000
        )
        assert cfg.datasets["dataset1"]["number_of_classes_per_attribute"]["label_"] == 1000


def test_s3_subsampled_datasets_register() -> None:
    """Test creation of config based on Yaml."""
    with hydra.initialize(version_base="1.1", config_path="_test-conf"):
        cfg = hydra.compose(config_name="test_s3_subsampled_datasets")
        assert (
            cfg["datasets"]["dataset1"]["_target_"]
            == "ood_inspector.datasets.dataset.InspectorDataset"
        )
        assert (
            cfg["datasets"]["dataset1"]["dataset"]["_target_"]
            == "ood_inspector.datasets.webdataset.get_fewshot_subsampled_dataset"
        )


def test_s3_subsampled_datasets_convenience_register() -> None:
    """Test creation of config based on Yaml."""
    with hydra.initialize(version_base="1.1", config_path="_test-conf"):
        cfg = hydra.compose(config_name="test_s3_subsampled_datasets_convenience")
        assert (
            cfg["datasets"]["dataset1"]["_target_"]
            == "ood_inspector.datasets.dataset.InspectorDataset"
        )
        assert (
            cfg["adaptation"]["dataset"]["dataset"]["_target_"]
            == "ood_inspector.datasets.webdataset.get_fewshot_subsampled_dataset"
        )
