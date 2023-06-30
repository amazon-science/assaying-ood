import unittest.mock as mock

import pytest

try:
    from ood_inspector.models import torchvision
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_pretrained_torchvision_model():
    assert torchvision.TorchvisionModelName.resnet18
    with pytest.raises(AttributeError):
        torchvision.TorchvisionModelName.doesnotexist


def test_torchvision_model():
    model = mock.MagicMock()
    assert torchvision.TorchvisionModel(model)
