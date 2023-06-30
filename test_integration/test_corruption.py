import pytest

try:
    import torchvision.datasets as tv_datasets
    from torchvision.transforms import transforms as tv_transforms

    import ood_inspector.corruption
    from ood_inspector.datasets.dataset import InspectorDataset
    from ood_inspector.datasets.transformation_stack import TransformationStack
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)

PRE_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
    ]
)


POST_TRANSFORM = tv_transforms.Normalize(mean=0.5, std=1.5)


def test_imagenet_c_type_corruption():
    # This corruption type uses the corruptions from the package `imagenet_c`.
    # Therefore this is listed as integration test.
    transform_stack = TransformationStack(
        tv_transforms.Compose([PRE_TRANSFORM, POST_TRANSFORM]), None
    )
    test_dataset = InspectorDataset(
        dataset=tv_datasets.FakeData(size=5),
        number_of_classes_per_attribute={"class": 5},
        default_attribute="class",
        transformations=transform_stack,
    )

    corruption = ood_inspector.corruption.ImageNetCTypeCorruption(
        datasets={"FakeData": test_dataset}, combine_corruption_types=True
    )
    corrupted_datasets = corruption.apply_corruptions_to_datasets()

    assert len(corruption.corruption_types) == 17  # number_of_corruptions.
    assert len(corruption.corruption_severities) == 5  # number_of_severities.
    # Since combine_corruption_types = True, this is also number_of_severities.
    assert len(corrupted_datasets) == 5
    # Expected total size = number samples * number corruptions
    expected_dataset_size = 5 * 17
    for severity in corruption.corruption_severities:
        key = f"Corrupted_FakeData_severity_{severity}"
        assert key in corrupted_datasets
        assert len(corrupted_datasets[key]) == expected_dataset_size


@pytest.mark.parametrize(
    "model_name",
    ["timm_pretrained_vgg11", "torchvision_pretrained_vgg11"],
)
def test_cli_imagenet_c_type_corruption_on_s3_dataset(
    cli_runner, s3dataset: str, model_name: str
) -> None:
    cli_runner(
        [
            "evaluations=classification_accuracy",
            f"+model={model_name}",
            f"+datasets={s3dataset}",
            "corruption=imagenet_c_type",
            "+corruption.corruption_types=[gaussian_noise, brightness]",
            "+corruption.corruption_severities=[1]",
            "corruption.combine_corruption_types=True",
        ],
        [
            "classification_accuracy",
            s3dataset,
            f"Corrupted_{s3dataset}_severity_1",
        ],
    )
