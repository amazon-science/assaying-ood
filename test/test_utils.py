import unittest.mock as mock
from unittest import TestCase

import numpy as np
import pytest

try:
    from ood_inspector import utils
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


@pytest.mark.parametrize(
    "dict_to_process,dict_result",
    [
        ({}, {}),
        ({"k": "v"}, {"k": "v"}),
        ({"k": ["v0", "v1"]}, {"k/0": "v0", "k/1": "v1"}),
        ({"k": np.array([0, 1])}, {"k/0": "0", "k/1": "1"}),
        ({"k": np.array([[0, 1]])}, {"k/0/0": "0", "k/0/1": "1"}),
        (
            {"k": np.array([[0, 1], [2, 3]])},
            {"k/0/0": "0", "k/0/1": "1", "k/1/0": "2", "k/1/1": "3"},
        ),
        ({"k": {"k1": "v1"}}, {"k/k1": "v1"}),
    ],
)
def test_shapes_image_augmenter(dict_to_process: dict, dict_result: dict):
    TestCase().assertEqual(utils.flatten_dict_to_scoped_json(dict_to_process), dict_result)


@mock.patch("boto3.client")
def test_write_to_s3(mock_client):

    client_instance = mock_client.return_value

    utils.write_to_s3('{"accuracy": 0.523"}'.encode(), "s3://test-bucket/prefix", "01-03/xyz.json")

    client_instance.upload_fileobj.assert_called_with(
        mock.ANY, "test-bucket", "prefix/01-03/xyz.json"
    )
