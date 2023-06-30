import enum

import pytest
import torch

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

mock_corruption_names = " ".join([f"corruption_{i}" for i in range(10)])
MOCK_CORRUPTION_TYPES = enum.Enum("MockCorruptionType", mock_corruption_names)


def get_fake_dataset():
    transform_stack = TransformationStack(
        tv_transforms.Compose([PRE_TRANSFORM, POST_TRANSFORM]), None
    )
    return InspectorDataset(
        dataset=tv_datasets.FakeData(size=5),
        number_of_classes_per_attribute={"class": 5},
        default_attribute="class",
        transformations=transform_stack,
    )


class MockCorruption(ood_inspector.corruption.Corruption):
    def _corrupt_image(self, image: torch.Tensor, corruption_type: str, severity: int):
        """Apply identity transform."""
        del corruption_type
        del severity

        # Check if inputs are as expected.
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert torch.all(0 <= image) and torch.all(image <= 1)
        return image


@pytest.mark.parametrize("combine_corruption_types", [False, True])
def test_apply_corruptions_to_datasets(combine_corruption_types):
    test_dataset = get_fake_dataset()
    datasets = {"test_dataset": test_dataset}

    corruption_severities = list(range(1, 6))
    mock_corruption = MockCorruption(
        datasets, list(MOCK_CORRUPTION_TYPES), corruption_severities, combine_corruption_types
    )

    corrupted_dataset = mock_corruption.apply_corruptions_to_datasets()

    if combine_corruption_types:
        assert len(corrupted_dataset) == len(corruption_severities)
        # Expected total size = number samples * number corruptions.
        expected_dataset_size = 5 * len(MOCK_CORRUPTION_TYPES)
        for severity in corruption_severities:
            key = f"Corrupted_test_dataset_severity_{severity}"
            assert key in corrupted_dataset
            assert len(corrupted_dataset[key]) == expected_dataset_size
    else:
        assert len(corrupted_dataset) == len(corruption_severities) * len(MOCK_CORRUPTION_TYPES)
        # Expected total size = number samples.
        expected_dataset_size = 5
        for severity in corruption_severities:
            for corruption_type in MOCK_CORRUPTION_TYPES:
                key = f"Corrupted_test_dataset_{corruption_type.name}_severity_{severity}"
                assert key in corrupted_dataset
                assert len(corrupted_dataset[key]) == expected_dataset_size


def test_no_corruption():
    fake_dataset = get_fake_dataset()
    corruption = ood_inspector.corruption.NoCorruption()
    corrupted_dataset = corruption.apply_corruptions_to_datasets(
        {"FakeData": fake_dataset}, PRE_TRANSFORM, POST_TRANSFORM
    )
    assert len(corrupted_dataset) == 0


def test_infer_pre_and_post_transform_with_final_normalization():
    transform = tv_transforms.Compose(
        [
            tv_transforms.Resize(256),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=0.5, std=1.5),
        ]
    )
    transformation_stack = TransformationStack(transform, None)
    pre_transform, normalization, post_transform = (
        transformation_stack.pre_normalization,
        transformation_stack.normalization,
        transformation_stack.post_normalization,
    )

    assert len(pre_transform.transforms) == 2
    assert len(post_transform.transforms) == 0
    assert isinstance(pre_transform.transforms[0], tv_transforms.Resize)
    assert isinstance(pre_transform.transforms[1], tv_transforms.ToTensor)
    assert isinstance(normalization, tv_transforms.Normalize)
