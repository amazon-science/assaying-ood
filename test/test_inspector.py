from unittest import mock

import pytest

try:
    from torchvision import models as tv_models

    from ood_inspector import inspector
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_inspector_run_result_is_correct() -> None:
    model = mock.MagicMock()
    model.eval.return_value = None
    evaluation = mock.MagicMock()
    evaluation.setup.return_value = None
    evaluation.update.return_value = None
    evaluation.score.return_value = "my_unique_value"
    requires_data_mock = mock.PropertyMock(return_value=False)
    type(evaluation).requires_data = requires_data_mock
    runner = inspector.Inspector(s3_output_path=None, model=model, evaluations={"mock": evaluation})
    results = runner.run()
    expected_results = {"mock": "my_unique_value"}
    assert results == expected_results
    # Called once on construction and once on eval.
    requires_data_mock.assert_called()
    model.eval.assert_called_once()


class FakeEval:
    def __init__(self, value):
        self.value = value

    @property
    def requires_data(self):
        return False


def test_save_and_load(tmpdir) -> None:
    vgg11 = tv_models.vgg11(pretrained=False)
    fake_evaluation = FakeEval("unique_value")

    saved_inspector = inspector.Inspector(
        model=vgg11, evaluations={"foo": fake_evaluation}, save_inspector=True
    )
    mock_config = {"config": 2}
    saved_inspector.save(tmpdir, config=mock_config)
    loaded_inspector = inspector.load(tmpdir)
    assert type(loaded_inspector.model) == tv_models.vgg.VGG
    assert loaded_inspector.evaluations["foo"].value == "unique_value"
