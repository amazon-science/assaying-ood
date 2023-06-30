import pytest


def test_cli_adaptation_no_adaptation(cli_runner, s3dataset: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            "+model=torchvision_pretrained_vgg11",
            f"+datasets={s3dataset}",
            "dataloader=TorchDataLoader",
            "adaptation=no_adaptation",
        ],
        ["classification_accuracy"],
    )


@pytest.mark.parametrize(
    "model",
    ["timm_pretrained_resnet18", "torchvision_pretrained_vgg11", "torchvision_pretrained_resnet18"],
)
def test_cli_adaptation_finetune(cli_runner, s3dataset: str, model: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"+model={model}",
            f"+datasets={s3dataset}",
            "dataloader=TorchDataLoader",
            "adaptation=finetune",
            "+adaptation.optimizer.defaults.lr=0.001",
            f"+adaptation.dataset={s3dataset}",
            "adaptation.number_of_epochs=2",
        ],
        ["Epoch 0/1", "Epoch 1/1", "Training complete", "classification_accuracy"],
    )


@pytest.mark.parametrize(
    "model",
    ["timm_pretrained_deit_base_distilled_patch16_224"],
)
def test_cli_adaptation_finetune_with_multiple_heads(
    cli_runner, s3dataset: str, model: str
) -> None:
    test_cli_adaptation_finetune(cli_runner, s3dataset, model)


@pytest.mark.parametrize(
    "model",
    ["timm_pretrained_resnet18"],
)
def test_cli_adaptation_finetune_with_lr_scheduler(cli_runner, s3dataset: str, model: str) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"+model={model}",
            f"+datasets={s3dataset}",
            "dataloader=TorchDataLoader",
            "adaptation=finetune",
            "+adaptation.optimizer.defaults.lr=1.0",
            f"+adaptation.dataset={s3dataset}",
            "adaptation.number_of_epochs=5",
            "adaptation/lr_scheduler=torch",
            "adaptation.lr_scheduler.classname=MultiStepLR",
            "+adaptation.lr_scheduler.options.milestones=[1,2,3]",
        ],
        [
            "Learning Rate 1",
            "Learning Rate 0.1",
            "Learning Rate 0.01",
            "Training complete",
            "classification_accuracy",
        ],
    )


@pytest.mark.parametrize(
    "augmenter",
    ["blur", "color_jitter", "random_color", "automix", "random_augment", "auto_augment", "augmix"],
)
def test_cli_adaptation_finetune_with_augmenter(cli_runner, s3dataset: str, augmenter: str) -> None:
    model = "timm_pretrained_resnet18"
    commands = [
        "evaluations=classification_accuracy",
        f"+model={model}",
        f"+datasets={s3dataset}",
        "dataloader=TorchDataLoader",
        "adaptation=finetune",
        f"+adaptation.dataset={s3dataset}",
        "adaptation.number_of_epochs=2",
        f"+adaptation.dataset.transformations.augmenter={augmenter}",
        "++adaptation.optimizer.defaults.lr=0.005",
    ]
    if augmenter == "color_jitter":
        commands.append("++adaptation.dataset.transformations.augmenter.brightness_factor=5")
    elif augmenter == "blur":
        commands.append("++adaptation.dataset.transformations.augmenter.radius=2.5")
    elif augmenter == "automix":
        commands.append(
            "+adaptation.dataset.transformations.augmenter.config_str='augmix-m5-w4-d3'"
        )
    cli_runner(
        commands,
        ["Epoch 0/1", "Epoch 1/1", "Training complete", "classification_accuracy"],
    )
