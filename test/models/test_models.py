import pytest
import torch
from torch import nn

try:
    from ood_inspector.models import inspector_base
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_number_of_parameters_is_correct():
    class TwoLayerNetwork(nn.Module):
        def __init__(self):
            super(TwoLayerNetwork, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, 3, bias=False)
            self.conv2 = nn.Conv2d(10, 2, 1, bias=False)

    inspector_base.InspectorModel.__abstractmethods__ = set()
    model = inspector_base.InspectorModel(TwoLayerNetwork())
    assert model.number_of_parameters == 10 * 3**2 + 10 * 2


def test_inspector_model_output_slicing():
    logits = torch.tensor([0.0, 0.0, 1.0, 1.0])
    features = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    output = inspector_base.InspectorModelOutput(logits, features)
    output_slice_1 = output[0:2]
    output_slice_2 = output[2:4]
    assert torch.all(output_slice_1.logits == logits[0:2])
    assert torch.all(output_slice_2.logits == logits[2:4])
    assert torch.all(output_slice_1.features == features[0:2])
    assert torch.all(output_slice_2.features == features[2:4])

    # Test case where features are None
    output = inspector_base.InspectorModelOutput(logits, features=None)
    output_slice_1 = output[0:2]
    assert output_slice_1.features is None
