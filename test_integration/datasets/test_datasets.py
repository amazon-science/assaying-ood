import pytest


def test_cli_multiple_datasets(cli_runner, s3dataset: str, s3_fairness_dataset: str) -> None:
    cli_runner(
        [
            "evaluations=[number_of_parameters,classification_accuracy]",
            "+model=torchvision_pretrained_vgg11",
            f"+datasets=[{s3dataset},{s3_fairness_dataset}]",
        ],
        [
            "classification_accuracy",
            "number_of_parameters",
            s3dataset,
            s3_fairness_dataset,
        ],
    )


@pytest.mark.skipif(True, reason="No write access to S3.")
def test_fewshot_subsampled_datasets(cli_runner):
    cli_runner(
        [
            "--config-dir=config_files/examples",
            "--config-name=example_fewshot_ds",
        ],
        [
            "Generating fewshot dataset - Min. class: 1/1, Max. class: 1/1, Seen classes: 1000/1000"
            "classification_accuracy",
            "Saving results",
            # TODO(zietld): Add a test that randomly generates a 'subsampling instance'
            # and check that resulting number of datapoints is correct.
        ],
    )


def test_fewshot_subsampled_datasets_cached(cli_runner):
    cli_runner(
        [
            "--config-dir=config_files/examples",
            "--config-name=example_fewshot_ds",
            "datasets.dataset1.dataset.force_create=False",
        ],
        [
            "classification_accuracy",
            "Saving results",
            "1000"  # size of subsampled dataset
            # TODO(zietld): Add a test that randomly generates a 'subsampling instance'
            # and check that resulting number of datapoints is correct.
        ],
    )
