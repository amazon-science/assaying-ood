import pytest

try:
    from ood_inspector.models import inspector_base
    from ood_inspector.models import mock as mockmodel
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_mock_model_forward():
    model = mockmodel.MockModel(device="cpu")
    output = model.forward()
    assert isinstance(output, inspector_base.InspectorModelOutput)
