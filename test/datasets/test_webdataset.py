"""Tests for Inspector datasets."""

import pytest

try:
    from ood_inspector.datasets import webdataset
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_get_webdataset():
    assert (
        webdataset.get_webdataset(
            uri_expression="s3://bucket_name/dataset.tar -",
            number_of_datapoints=1,
            number_of_classes_per_attribute={"class": 100},
        )
        is not None
    )


def test_get_webdataset_multiple_attributes():
    assert (
        webdataset.get_webdataset(
            uri_expression="s3://bucket_name/dataset.tar -",
            number_of_datapoints=1000,
            number_of_classes_per_attribute={"gender": 2, "race": 4},
        )
        is not None
    )
