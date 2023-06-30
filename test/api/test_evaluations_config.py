import hydra.utils
import pytest

try:
    from ood_inspector import evaluations  # noqa: F401
    from ood_inspector.api import evaluations_config
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_mock_evaluation_config() -> None:
    assert evaluations_config.MockEvaluationConfig() is not None
    assert hydra.utils.instantiate(evaluations_config.MockEvaluationConfig()) is not None


def test_accuracy_evaluation_config() -> None:
    assert evaluations_config.ClassificationAccuracyConfig() is not None
    assert hydra.utils.instantiate(evaluations_config.ClassificationAccuracyConfig()) is not None


def test_accuracy_per_group_evaluation_config() -> None:
    assert evaluations_config.ClassificationAccuracyPerGroupConfig() is not None
    assert (
        hydra.utils.instantiate(evaluations_config.ClassificationAccuracyPerGroupConfig())
        is not None
    )


def test_expected_calibration_error_evaluation_config() -> None:
    assert evaluations_config.ExpectedCalibrationErrorConfig() is not None
    assert hydra.utils.instantiate(evaluations_config.ExpectedCalibrationErrorConfig()) is not None


def test_negative_log_likelihood_evaluation_config() -> None:
    assert evaluations_config.NegativeLogLikelihoodConfig() is not None
    assert hydra.utils.instantiate(evaluations_config.NegativeLogLikelihoodConfig()) is not None


def test_demographic_parity_config() -> None:
    with hydra.initialize_config_module(version_base="1.1", config_module="ood_inspector"):
        cfg = hydra.compose(
            config_name="inspector",
            overrides=["evaluations=[demographic_parity_inferred_groups]"],
        )
        assert evaluations_config.DemographicParityInferredGroupsConfig() is not None
        assert (
            hydra.utils.instantiate(cfg.evaluations["demographic_parity_inferred_groups"])
            is not None
        )


def test_output_distribution_per_group_evaluation_config() -> None:
    assert evaluations_config.OutputDistributionPerGroupConfig() is not None
    assert (
        hydra.utils.instantiate(evaluations_config.OutputDistributionPerGroupConfig()) is not None
    )
