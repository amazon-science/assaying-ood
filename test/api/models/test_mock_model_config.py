import pytest

try:
    from ood_inspector.api.models import mock_config
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


@pytest.mark.parametrize("pretrained", [True, False])
def test_mock_model_config(pretrained):
    mock_config.MockModelConfig("ignore_this_model_name", pretrained)
