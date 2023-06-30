import os

import pytest


@pytest.mark.parametrize("multirun,expected_dirname", [(True, "multirun"), (False, "outputs")])
def test_cli_savedir(cli_runner, multirun: bool, expected_dirname: str) -> None:
    arguments = [
        "evaluations=number_of_parameters",
        "+model=torchvision_pretrained_vgg11",
        "s3_output_path=s3://test",
        "save_inspector=True",
    ]

    if multirun:
        arguments.append("-m")

    # Command returns non-zero error code as it fails during saving to s3 (the bucket does not
    # exist).
    cli_runner(
        arguments,
        [
            f"Saving run outputs to s3 path s3://test/{expected_dirname}",
        ],
        expect_successful=False,
    )


def test_plugins_loader(cli_runner, s3dataset: str) -> None:
    os.environ["INSPECTOR_CUSTOM_MODULES_PATH"] = os.path.join(
        os.path.dirname(__file__), "../custom_modules"
    )

    arguments = [
        "+model=fcnet",
        "evaluations=custom_metric",
        f"+datasets={s3dataset}",
    ]

    cli_runner(
        arguments,
        ["Found user module: custom_model", "Found user module: custom_metric"],
        expect_successful=True,
    )
