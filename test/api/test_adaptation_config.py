import pytest

try:
    from ood_inspector.api import adaptation_config
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_no_adaptation_config() -> None:
    assert adaptation_config.NoAdaptationConfig() is not None
