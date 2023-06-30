import pytest
import torch


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11", "timm_pretrained_resnet18", "torchvision_pretrained_resnet18"],
)
@pytest.mark.parametrize("top_k", [1, 3])
def test_cli_accuracy(cli_runner, s3dataset: str, model_name: str, top_k: int) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"++evaluations.classification_accuracy.top_k={top_k}",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
        ],
        ["classification_accuracy"],
    )


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11", "timm_pretrained_resnet18", "torchvision_pretrained_resnet18"],
)
@pytest.mark.parametrize("top_k", [1, 3])
def test_cli_accuracy_per_group(cli_runner, s3dataset: str, model_name: str, top_k: int) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy_per_group",
            f"++evaluations.classification_accuracy_per_group.top_k={top_k}",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
        ],
        ["classification_accuracy_per_group"],
    )


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11", "timm_pretrained_resnet18", "torchvision_pretrained_resnet18"],
)
def test_cli_output_distribution_per_group(cli_runner, s3dataset: str, model_name: str) -> None:
    cli_runner(
        [
            "evaluations=output_distribution_per_group",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
        ],
        ["output_distribution_per_group"],
    )


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11", "timm_pretrained_resnet18", "torchvision_pretrained_resnet18"],
)
@pytest.mark.parametrize("device_name", ["cuda", "cpu"])
def test_cli_accuracy_on_device(
    cli_runner, s3dataset: str, model_name: str, device_name: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        return
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
            f"++model.device={device_name}",
        ],
        ["classification_accuracy"],
    )


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11", "torchvision_pretrained_resnet18"],
)
@pytest.mark.parametrize("attack_name", ["FGSM", "PGD"])
def test_cli_adversarial(cli_runner, s3dataset: str, model_name: str, attack_name: str) -> None:
    cli_runner(
        [
            "evaluations=adversarial_accuracy",
            f"+evaluations.adversarial_accuracy.attack.name={attack_name}",
            "++evaluations.adversarial_accuracy.attack.random_start=True",
            "+evaluations.adversarial_accuracy.attack_sizes=[0., .1]",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
        ],
        ["adversarial_accuracy"],
    )


@pytest.mark.parametrize(
    "attack_config_name",
    ["linf_auto_attack", "l2_auto_attack", "linf_custom_auto_attack", "linf_apgd_dlr_auto_attack"],
)
def test_cli_autoattack(cli_runner, s3dataset: str, attack_config_name: str) -> None:
    cli_runner(
        [
            f"evaluations={attack_config_name}",
            f"+evaluations.{attack_config_name}.attack_size=0.015",
            "+model=torchvision_pretrained_resnet18",
            f"+datasets={s3dataset}",
        ],
        [f"{attack_config_name}"],
    )


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_deit_base_distilled_patch16_224"],
)
def test_cli_adversarial_with_multiple_heads(cli_runner, s3dataset: str, model_name: str) -> None:
    test_cli_adversarial(cli_runner, s3dataset, model_name, "FGSM")


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11"],
)
@pytest.mark.parametrize("number_bins", [10, 15])
def test_cli_ece(cli_runner, s3dataset: str, model_name: str, number_bins: int) -> None:
    cli_runner(
        [
            "evaluations=expected_calibration_error",
            f"++evaluations.expected_calibration_error.number_bins={number_bins}",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
        ],
        ["expected_calibration_error"],
    )


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11"],
)
@pytest.mark.parametrize("device_name", ["cuda", "cpu"])
def test_cli_ece_on_device(cli_runner, s3dataset: str, model_name: str, device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        return
    cli_runner(
        [
            "evaluations=expected_calibration_error",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
            f"++model.device={device_name}",
        ],
        ["expected_calibration_error"],
    )


def test_cli_number_of_parameters_with_timm_model(cli_runner) -> None:
    cli_runner(
        [
            "evaluations=number_of_parameters",
            "+model=timm_pretrained_vgg11",
        ],
        ["Loading pretrained weights", "number_of_parameters", "132863336"],
    )


def test_cli_number_of_parameters_with_torchvision_model(cli_runner) -> None:
    cli_runner(
        [
            "evaluations=number_of_parameters",
            "+model=torchvision_pretrained_resnet18",
        ],
        ["number_of_parameters", "11689512"],
    )


def test_cli_multiple_evaluations(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=[number_of_parameters,classification_accuracy]",
            "+model=torchvision_pretrained_vgg11",
            f"+datasets={s3dataset}",
        ],
        ["classification_accuracy", "number_of_parameters"],
    )


def test_cli_demographic_parity(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=demographic_parity_inferred_groups",
            "++evaluations.demographic_parity_inferred_groups.number_of_samples=6",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
        ],
        ["demographic_parity_inferred_groups"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU, torch and CUDA.")
def test_cli_demographic_parity_on_cuda(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=demographic_parity_inferred_groups",
            "++evaluations.demographic_parity_inferred_groups.number_of_samples=6",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
            "++model.device=cuda",
        ],
        ["demographic_parity_inferred_groups"],
    )


def test_cli_negative_log_likelihood(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=negative_log_likelihood",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
        ],
        ["negative_log_likelihood"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU, torch and CUDA.")
def test_cli_negative_log_likelihood_on_cuda(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=negative_log_likelihood",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
            "++model.device=cuda",
        ],
        ["negative_log_likelihood"],
    )


def test_cli_raises_if_metric_requires_data_but_none_given(cli_runner) -> None:
    cli_runner(
        [
            "evaluations=[number_of_parameters,classification_accuracy]",
            "+model=torchvision_pretrained_vgg11",
        ],
        ["ValueError: Evaluation classification_accuracy"],
        # While we expect this run to fail, it looks like hydra does not pass the return value and
        # always returns 0 (i.e. successful execution).  Thus unintuitively, this run is considered
        # successful.
        expect_successful=True,
    )
