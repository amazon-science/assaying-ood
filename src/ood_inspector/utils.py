"""Utility functions."""
from __future__ import annotations

import functools
import itertools
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
from botocore.exceptions import ClientError
from torch import nn


class TensorTracker:
    """A wrapper class which tracks a tensor.

    We need this in order to pass around inplace modifications.  There is no
    other way how we can extract information from a hook.
    """

    def __init__(self, initial_values):
        self.values = initial_values


def track_input_hook(tracker, _layer, inputs):
    """Track the input of a layer via forward pre hook."""
    tracker.values = inputs


def track_input_of_layer(
    layer: nn.Module,
) -> Tuple[torch.utils.hooks.RemovableHandle, TensorTracker]:
    """Track the input of a layer during forward passes.

    We use a forward pass hook for this in order to avoid the necessity of
    changing the model code.

    Args:
        layer: nn.Module of which to track the inputs to the forward call.

    Returns:
        hook_handle: Can be used to remove the hook.
        FeatureTracker: Object which holds a reference to the most recent input
        to the layer.
    """
    tracker = TensorTracker(None)

    # Use partial instead of local function to allow pickling.
    handle = layer.register_forward_pre_hook(functools.partial(track_input_hook, tracker))
    return handle, tracker


def get_device_from_module(module: nn.Module) -> torch.device:
    """Gets the device a module is located on.

    This assumes that the whole module is on the same device, and thus only considers the first
    parameter or buffer of the module to extract the device.

    Args:
        module (nn.Module): The module of which we should get the device.

    Returns:
        torch.device: The device on which module is located.
    """

    # Build an iterator over parameters and buffers. Thus if a module does not have any parameters,
    # the buffers are also considered.
    return next(itertools.chain(module.parameters(), module.buffers())).device


def flatten_dict_to_scoped_json(dict_to_flatten, separator="/"):
    result_dict = {}

    def flatten(current_item, new_key=""):
        if new_key == "":
            prefix = ""
        else:
            prefix = new_key + separator
        if type(current_item) is dict:
            for key, value in current_item.items():
                flatten(value, prefix + str(key))
        elif type(current_item) is list:
            for index, value in enumerate(current_item):
                flatten(value, prefix + str(index))
        elif type(current_item) is np.ndarray:
            for index, value in enumerate(current_item):
                flatten(value, prefix + str(index))
        else:
            result_dict[new_key] = str(current_item)

    flatten(dict_to_flatten)
    return result_dict


def strip_s3_url(s3path):
    """Strips a full s3 utl in bucket name and path to node

    Example: path="s3://bucket_name/path/to/node/"
    Returns: "bucket_name", "path/to/node"
    """
    parsed_path = urlparse(s3path, allow_fragments=False)
    bucket = parsed_path.netloc
    path = parsed_path.path.strip("/")
    return bucket, path


def write_to_s3(json_dump: str, out_path: str, filename: str):
    bucket, prefix = strip_s3_url(out_path)
    s3 = boto3.client("s3")
    key = prefix + "/" + filename
    try:
        with BytesIO(json_dump) as json_obj:
            s3.upload_fileobj(json_obj, bucket, key)
    except Exception as e:
        logging.error(f"Write to s3 failed with: {e}")


def flush_logs():
    """Flush logs to disc."""
    for handler in logging.getLogger().handlers:
        handler.flush()


def copy_run_output_to_s3(s3_output_path):
    """Copy all files in the current directory to s3_output_path."""
    paths = []

    for file_path in Path(".").rglob("*"):
        if file_path.is_dir():
            continue
        str_file_path = str(file_path)
        paths.append(str_file_path)

    bucket, prefix = strip_s3_url(s3_output_path)

    s3_client = boto3.client("s3")
    for path in paths:
        try:
            s3_client.upload_file(path, bucket, os.path.join(prefix, path))
        except ClientError as e:
            logging.error("An error occurred while saving to s3.")
            logging.exception(e)


def s3_list(path):
    s3 = boto3.resource("s3")
    bucket, path = strip_s3_url(path)
    if not path[-1] == "/":
        path += "/"
    my_bucket = s3.Bucket(bucket)
    objects = [
        f"s3://{bucket}/{object_summary.key}"
        for object_summary in my_bucket.objects.filter(Prefix=path)
    ]
    return objects


def s3_folder_exists_and_not_empty(path):
    bucket, path = strip_s3_url(path)
    if not path[-1] == "/":
        path += "/"
    s3 = boto3.client("s3")
    response = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter="/", MaxKeys=1)
    return "Contents" in response
