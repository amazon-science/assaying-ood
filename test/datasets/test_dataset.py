import pytest

try:
    import torchvision.datasets as tv_datasets

    from ood_inspector.datasets.dataset import InspectorDataset

except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_dataset_instantiation_dict_number_of_classes():
    dataset = InspectorDataset(
        dataset=tv_datasets.FakeData(size=5),
        number_of_classes_per_attribute={"label_": 5, "another_label": 10},
        default_attribute="label_",
    )
    assert dataset.number_of_classes_per_attribute == {"label_": 5, "another_label": 10}


def test_dataset_instantiation_int_number_of_classes():
    dataset = InspectorDataset(
        dataset=tv_datasets.FakeData(size=5),
        number_of_classes_per_attribute=10,
    )
    assert dataset.number_of_classes_per_attribute == {"label_": 10}
