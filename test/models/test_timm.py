import unittest.mock as mock

import pytest

try:
    from ood_inspector.models import timm
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_pretrained_timm_model():
    assert timm.TimmModelName.resnet18
    with pytest.raises(AttributeError):
        timm.TimmModelName.doesnotexist


def test_timm_model():
    timm_model = mock.MagicMock()
    nn_layer = mock.MagicMock()
    timm_model.default_cfg = {"classifier": "head", "num_classes": 1}
    timm_model.head = nn_layer
    assert timm.TimmModel(timm_model)
