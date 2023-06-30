from unittest import mock

import pytest

try:
    import ood_inspector.adaptation
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_no_adaptation() -> None:
    model = mock.MagicMock()
    no_adaptation = ood_inspector.adaptation.NoAdaptation()
    assert model is no_adaptation.fit(model)
