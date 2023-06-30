from typing import Dict, List

import pytest


def test_cli_torchdataloader(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
            "dataloader=TorchDataLoader",
        ],
        ["classification_accuracy", "Loading pretrained weights"],
    )


def test_cli_dataloader_batch_size(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
            "dataloader.batch_size=2",
        ],
        ["classification_accuracy"],
    )


@pytest.mark.parametrize(
    "classname,options,number_of_epochs,outputs",
    [
        (
            "CosineAnnealingLR",
            {"T_max": 4, "eta_min": 0.1},
            5,
            [1, 0.86819, 0.55, 0.23180, 0.1],
        ),
        (
            "CosineAnnealingWarmRestarts",
            {"T_0": 4, "T_mult": 1, "eta_min": 0.1},
            6,
            [1, 0.86819, 0.55, 0.23180, 1],
        ),
        (
            "ExponentialLR",
            {"gamma": 0.5},
            3,
            [1, 0.5, 0.25],
        ),
        (
            "MultiStepLR",
            {"milestones": "[1,2]", "gamma": 0.4},
            3,
            [1, 0.4, 0.16],
        ),
        (
            "StepLR",
            {"step_size": 2, "gamma": 0.5},
            5,
            [1, 0.5, 0.25],
        ),
        (
            "StepLR",
            {"step_size": 2, "invalid_argument_name": 0.5},
            5,
            ["got an unexpected keyword argument 'invalid_argument_name'"],
        ),
        (
            "StepLR",
            {},
            5,
            ["missing", "required positional argument"],
        ),
        (
            None,
            {},
            1,
            ["No learning rate scheduler used."],
        ),
    ],
)
def test_cli_lr_scheduler(
    cli_runner,
    s3dataset: str,
    classname: str,
    options: Dict,
    number_of_epochs: int,
    outputs: List,
) -> None:
    arguments = [
        "evaluations=classification_accuracy",
        "+model=timm_pretrained_vgg11",
        f"+datasets={s3dataset}",
        "dataloader=TorchDataLoader",
        "adaptation=finetune",
        "+adaptation.optimizer.defaults.lr=1.0",
        f"+adaptation.dataset={s3dataset}",
        f"adaptation.number_of_epochs={number_of_epochs}",
    ]

    if classname:
        arguments += [
            "adaptation/lr_scheduler=torch",
            f"adaptation.lr_scheduler.classname={classname}",
        ]

    for key, value in options.items():
        arguments.append(
            f"+adaptation.lr_scheduler.options.{key}={value}",
        )

    expected_outputs = []
    for output in outputs:
        if isinstance(output, str):
            expected_outputs.append(output)
        else:
            expected_outputs.append(f"Learning Rate {output}")

    cli_runner(arguments, expected_outputs)


def test_cli_vtab_lr_scheduler(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "+model=timm_pretrained_vgg11",
            f"+datasets={s3dataset}",
            "dataloader=TorchDataLoader",
            "adaptation=finetune",
            "+adaptation.optimizer.defaults.lr=1.0",
            f"+adaptation.dataset={s3dataset}",
            f"adaptation.number_of_epochs={5}",
            "adaptation/lr_scheduler=multistep_30_60_90",
        ],
        ["MultiStepLR"] + [f"Learning Rate {lr}" for lr in [1.0, 0.1, 0.01]],
    )
