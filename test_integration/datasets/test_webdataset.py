import pytest

try:
    from ood_inspector.datasets import webdataset
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_get_webdataset():
    dataset = webdataset.get_webdataset(
        uri_expression="s3://inspector-fairness-datasets/sharded/fairface/"
        "TinyFairFaceForTesting/shard-000000.tar",
        number_of_datapoints=1000,
        number_of_classes_per_attribute={"age": 7, "gender": 2, "race": 7, "service_test": 2},
    )
    assert len(dataset) == 1000
    assert {"age", "gender", "race", "service_test"}.issubset(set(next(iter(dataset)).keys()))


@pytest.mark.parametrize(
    "path,expected_size",
    [
        ("UTKFace/aligned-and-cropped/shard-{000000..000002}.tar", 23704),
    ],
)
def test_loading_fairness_dataset(path: str, expected_size: int) -> None:
    dataset = webdataset.get_webdataset(
        uri_expression="s3://inspector-fairness-datasets/sharded/" + f"{path}",
    )
    assert len(dataset) == expected_size
    assert {"age", "date_and_time", "gender", "race"}.issubset(set(next(iter(dataset)).keys()))


@pytest.mark.parametrize(
    "fairness_metric", ["classification_accuracy_per_group", "output_distribution_per_group"]
)
def test_cli_evaluation_on_s3_fairness_dataset(
    cli_runner,
    s3_fairness_dataset: str,
    fairness_metric: str,
    eval_target_attribute: str,
    eval_group_attribute: str,
) -> None:
    cli_runner(
        [
            f"evaluations={fairness_metric}",
            f"evaluations.{fairness_metric}.target_attribute=" f"{eval_target_attribute}",
            f"evaluations.{fairness_metric}.group_attribute={eval_group_attribute}",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3_fairness_dataset}",
        ],
        [fairness_metric],
    )


@pytest.mark.parametrize("s3_fairness_dataset", ["S3CelebA-val", "S3UTKFace-aligned-and-cropped"])
def test_cli_evaluation_on_s3_fairness_dataset_default_attribute(
    cli_runner, s3_fairness_dataset: str
) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3_fairness_dataset}",
        ],
        ["classification_accuracy"],
    )


def test_cli_evaluation_on_s3_fairness_dataset_not_existing_attribute(
    cli_runner, s3_fairness_dataset: str
) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "evaluations.classification_accuracy.target_attribute=I_DO_NOT_EXIST",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3_fairness_dataset}",
        ],
        ["classification_accuracy"],
    )


def test_cli_adaptation_on_s3_fairness_dataset(
    cli_runner,
    s3_fairness_dataset: str,
    eval_target_attribute: str,
    adaptation_target_attribute: str,
) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"evaluations.classification_accuracy.target_attribute={eval_target_attribute}",
            f"+datasets={s3_fairness_dataset}",
            "+model=torchvision_pretrained_vgg11",
            "adaptation=finetune",
            "adaptation.number_of_epochs=1",
            f"adaptation.target_attribute={adaptation_target_attribute}",
            f"+adaptation.dataset={s3_fairness_dataset}",
            "adaptation.finetune_only_head=True",
            "+adaptation.optimizer.defaults.lr=0.001",
        ],
        ["Training complete", "classification_accuracy"],
    )


def test_cli_adaptation_on_s3_fairness_dataset_default_attribute(
    cli_runner, s3_fairness_dataset: str
) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"+datasets={s3_fairness_dataset}",
            "+model=torchvision_pretrained_vgg11",
            "adaptation=finetune",
            "adaptation.number_of_epochs=1",
            f"+adaptation.dataset={s3_fairness_dataset}",
            "adaptation.finetune_only_head=True",
            "+adaptation.optimizer.defaults.lr=0.001",
        ],
        ["Training complete", "classification_accuracy"],
    )


def test_cli_evaluation_on_s3dataset(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
        ],
        ["classification_accuracy"],
    )


def test_cli_adaptation_on_s3_dataset(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"+datasets={s3dataset}",
            "+model=torchvision_pretrained_vgg11",
            "adaptation=finetune",
            "adaptation.number_of_epochs=1",
            f"+adaptation.dataset={s3dataset}",
            "adaptation.finetune_only_head=True",
            "+adaptation.optimizer.defaults.lr=0.001",
        ],
        ["Training complete", "classification_accuracy"],
    )
