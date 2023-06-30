import hydra.utils
import pytest

try:
    from ood_inspector import corruption  # noqa: F401
    from ood_inspector.api import corruption_config
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_no_corruption_config() -> None:
    assert corruption_config.NoCorruptionConfig() is not None
    assert hydra.utils.instantiate(corruption_config.NoCorruptionConfig()) is not None


def test_imagenet_c_type_corruption_config() -> None:
    assert corruption_config.ImageNetCTypeCorruptionConfig() is not None
    assert hydra.utils.instantiate(corruption_config.ImageNetCTypeCorruptionConfig()) is not None
