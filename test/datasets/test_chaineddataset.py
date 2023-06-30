"""Tests for Inspector datasets."""

import pytest

try:
    from torchvision.transforms import transforms as tv_transforms

    from ood_inspector.datasets import webdataset
    from ood_inspector.datasets.dataset import ChainedDataset, InspectorDataset
    from ood_inspector.datasets.transformation_stack import TransformationStack
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)

PRE_TRANSFORM_1 = tv_transforms.Compose(
    [
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
    ]
)
PRE_TRANSFORM_2 = tv_transforms.Compose(
    [
        tv_transforms.Resize(512),
        tv_transforms.CenterCrop(112),
        tv_transforms.ToTensor(),
    ]
)
POST_TRANSFORM_1 = tv_transforms.Normalize(mean=0.5, std=1.5)
POST_TRANSFORM_2 = tv_transforms.Normalize(mean=0.55, std=1.45)


def test_chained_dataset_uniqueness():
    dataset = webdataset.get_webdataset(
        uri_expression="s3://bucket_name/dataset.tar -",
        number_of_datapoints=1000,
        number_of_classes_per_attribute={"gender": 2, "race": 4},
    )
    transform_stack_11 = TransformationStack(
        tv_transforms.Compose([PRE_TRANSFORM_1, POST_TRANSFORM_1]), None
    )
    transform_stack_12 = TransformationStack(
        tv_transforms.Compose([PRE_TRANSFORM_1, POST_TRANSFORM_2]), None
    )
    transform_stack_21 = TransformationStack(
        tv_transforms.Compose([PRE_TRANSFORM_2, POST_TRANSFORM_1]), None
    )
    transform_stack_22 = TransformationStack(
        tv_transforms.Compose([PRE_TRANSFORM_2, POST_TRANSFORM_2]), None
    )
    # only second digit indicates matching normalization parameters
    dataset_11 = InspectorDataset(
        dataset=dataset,
        number_of_classes_per_attribute={"gender": 2, "race": 4},
        default_attribute="gender",
        transformations=transform_stack_11,
    )
    dataset_12 = InspectorDataset(
        dataset=dataset,
        number_of_classes_per_attribute={"gender": 2, "race": 4},
        default_attribute="gender",
        transformations=transform_stack_12,
    )
    dataset_21 = InspectorDataset(
        dataset=dataset,
        number_of_classes_per_attribute={"gender": 2, "race": 4},
        default_attribute="gender",
        transformations=transform_stack_21,
    )
    dataset_22 = InspectorDataset(
        dataset=dataset,
        number_of_classes_per_attribute={"gender": 2, "race": 4},
        default_attribute="gender",
        transformations=transform_stack_22,
    )

    ChainedDataset([dataset_11, dataset_11]).normalization
    ChainedDataset([dataset_11, dataset_21]).normalization
    ChainedDataset([dataset_12, dataset_12]).normalization
    ChainedDataset([dataset_12, dataset_22]).normalization
    ChainedDataset([dataset_21, dataset_21]).normalization
    ChainedDataset([dataset_22, dataset_22]).normalization

    with pytest.raises(ValueError):
        ChainedDataset([dataset_11, dataset_12]).normalization

    with pytest.raises(ValueError):
        ChainedDataset([dataset_22, dataset_11]).normalization

    with pytest.raises(ValueError):
        ChainedDataset([dataset_22, dataset_21]).normalization
