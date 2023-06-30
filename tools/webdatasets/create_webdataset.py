"""Turn a local dataset into a sharded S3 Webdataset and optionally split it.

This script assumes the data is given in the format
    {dataset}/{class}/*
or in the format
    {dataset}/{environment}/{class}/*
when using the `--recursive` option.

An example from DomainBed is
    VLCS/LabelMe/bird
    VLCS/LabelMe/car
    ...
    VLCS/Caltech101/bird
    VLCS/Caltech101/car
    ...

An example call to this function could be
    `python3 tools/create_webdataset.py /efs/domainbed/data/VLCS/LabelMe output/VLCS/LabelMe`
wich would generate shards in the format
    output/VLCS/LabelMe/shard-000000.tar
    output/VLCS/LabelMe/shard-000001.tar
    ...

An example for a **recursive** call could be
    `python3 tools/create_webdataset.py /efs/domainbed/data/VLCS output/VLCS --recursive`
which would generate shards in the format
    output/VLCS/LabelMe/shard-000000.tar
    output/VLCS/LabelMe/shard-000001.tar
    ...
    output/VLCS/Caltech101/shard-000000.tar
    ....
"""
import ast
import glob
import logging
import os.path
import random
from typing import Callable, Dict, Optional, Tuple

import click
import numpy as np
from PIL import Image
import torchvision.datasets as tv_datasets
import tqdm
import webdataset


class TupleType(click.ParamType):
    """Custom type Tuple for CLI inputs.

    Click package doesn't accept any tuple nor lists per default. With this custom TupleType, you
    can pass tuple inputs with the CLI. Tuples must be quoted, i.e. passed as strings, as in:
    `python3 create_valtest_splits.py SRC_DATASET TARGET_DATASET --split-ratios '(.8, .1, .1)'`
    """

    def __init__(self, name: str = "tuple", base_type: Optional[type] = None, **kwargs):
        super().__init__(**kwargs)
        self.base_type = base_type
        self.name = name

    def _is_correct_type(self, value):
        if isinstance(value, tuple):
            if (self.base_type is None) or (not value) or isinstance(value[0], self.base_type):
                return True
        return False

    def convert(self, value, param, ctx):
        if self._is_correct_type(value):
            return value

        try:
            converted_value = ast.literal_eval(value)
            if type(converted_value) == list:
                converted_value = tuple(converted_value)
            if not self._is_correct_type(converted_value):
                raise ValueError
            return converted_value
        except ValueError:
            type_string = ""
            if self.base_type is not None:
                type_string = f"of type {self.base_type}"
            self.fail(f"{value!r} is not a valid tuple" + type_string, param, ctx)


TUPLE_OF_FLOATS = TupleType(base_type=float)
TUPLE_OF_STRINGS = TupleType(base_type=str)


@click.command()
@click.argument("input_dir", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--maxcount", type=int, default=10000, help="Maximum number of elements per shard.")
@click.option(
    "--maxsize", type=int, default=int(3e8), help="Maximum size of shard in Bytes. Default 300MB."
)
@click.option("--shuffle", is_flag=True, help="Shuffle data before sharding.")
@click.option("--recursive", is_flag=True, help="Create a dataset for each dir in input_dir")
@click.option(
    "--split-ratios",
    type=TUPLE_OF_FLOATS,
    default=(1.0,),
    help="Split dataset in given proportions.",
)
@click.option(
    "--split-names",
    type=TUPLE_OF_STRINGS,
    default=("",),
    help="Split folder names. Use one unique name per split, e.g. ('train', 'val', 'test')",
)
def cli(
    input_dir: str,
    output_dir: str,
    maxcount: int,
    maxsize: int,
    shuffle: bool,
    recursive: bool,
    split_ratios: Tuple[float],
    split_names: Tuple[str],
):
    base_pattern: str = "shard-%06d.tar"
    if recursive:
        # If recursive, dataset_paths and dataset_names actually correspond to environments
        dataset_paths = [path for path in glob.glob(f"{input_dir}/*") if os.path.isdir(path)]
        logging.info(
            f"Found {len(dataset_paths)} environments in dataset {os.path.basename(input_dir)}"
        )
        dataset_names = [os.path.basename(dataset_path) for dataset_path in dataset_paths]
        output_paths = [os.path.join(output_dir, dataset_name) for dataset_name in dataset_names]
    else:
        dataset_paths = [input_dir]
        dataset_names = [os.path.basename(input_dir)]
        output_paths = [output_dir]

    if len(split_names) == 1 and len(split_ratios) > 1:
        split_names = [f"split-{i}" for i in range(len(split_ratios))]
    assert len(set(split_names)) == len(split_names) == len(split_ratios)

    # TODO(pgehler): support non-sharded webdataset.TarWriter
    def writer(pattern):
        return webdataset.ShardWriter(pattern, maxsize=maxsize, maxcount=maxcount, keep_meta=True)

    for dataset_path, dataset_name, output_path in zip(dataset_paths, dataset_names, output_paths):
        logging.info(f"Processing {dataset_name}")
        writers = list()
        for split_name in split_names:
            os.makedirs(os.path.join(output_path, split_name))
            writers.append(writer(os.path.join(output_path, split_name, base_pattern)))
        write_webdataset(
            dataset_path,
            tuple(writers),
            shuffle,
            split_ratios,
        )


def read_image(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()


def get_class_sizes(dataset_path: str) -> Dict[str, int]:
    """Returns a map from a class name (a string) to the size of the class."""
    class_sizes = dict()
    for class_name in os.listdir(dataset_path):
        path_to_class = os.path.join(dataset_path, class_name)
        if os.path.isdir(path_to_class):
            class_sizes[class_name] = len(
                [name for name in os.listdir(path_to_class)]
            )  # Assumes that path_to_class contains only files.
    return class_sizes


def get_class_split_sizes(
    class_sizes: Dict[str, int], split_ratios: Tuple[int], class_to_idx: Dict[str, int]
) -> Dict[int, Tuple[int]]:
    """Returns a map from a class label (an integer) to a tuple with the per-split data class sizes.

    The tuples all contain number_of_splits integers. The class labels are computed using the class
    names found in the keys of class_sizes, and class_to_idx, which maps class names to class
    labels.
    """
    splitted_class_sizes = {
        name: [int(split * size) for split in split_ratios] for name, size in class_sizes.items()
    }
    relative_rounding_errors = {
        name: [split * size - int(split * size) / (int(split * size) + 1) for split in split_ratios]
        for name, size in class_sizes.items()
    }
    decreasing_error_indices = {
        name: np.argsort(-np.array(errors), axis=-1)
        for name, errors in relative_rounding_errors.items()
    }

    # For each class, distribute the remaining datapoints among splits one by one, starting with the
    # splits that have the highest relative rounding error
    for name, decreasing_idx in decreasing_error_indices.items():
        for i in range(class_sizes[name] - sum(splitted_class_sizes[name])):
            splitted_class_sizes[name][decreasing_idx[i]] += 1
    output = {class_to_idx[name]: tuple(sizes) for name, sizes in splitted_class_sizes.items()}
    return output


def write_webdataset(
    dataset_path: str,
    writers: Tuple[Callable],
    shuffle: bool,
    split_ratios: Tuple = (1.0,),  # size of splits; must sum to 1.
):
    """
    Partitions the dataset into splits based on the given split_ratios and writes each split to a
    split-specific, sharded webdataset.

    To do so, this functions first computes the exact per-class split sizes based on the given split
    ratios. It then partitions the dataset into splits by sampling each datapoint one by one,
    checking its label, and allocating it to the first split that is not yet full for that class
    label.
    """

    dataset = tv_datasets.ImageFolder(dataset_path)
    class_to_idx = dataset.class_to_idx

    # Compute per class split sizes based on the split_ratios

    class_sizes = get_class_sizes(dataset_path)
    class_names = class_sizes.keys()
    dataset_class_names = set(dataset.classes)
    if sum(class_sizes.values()) != len(dataset):
        raise ValueError(
            f"ImageFolder dataset has {len(dataset)} datapoints, but found "
            f"{sum(class_sizes.values())} files or folders in folder {dataset_path}."
        )
    for class_name in class_names:
        if class_name not in dataset_class_names:
            raise ValueError(
                f"Found folder {class_name} in {dataset_path} which does not correspond to "
                f"a class name in the ImageFolder-dataset constructed using the previous path."
            )
        assert class_name in dataset_class_names
    if sum(split_ratios) != 1.0:
        raise ValueError("Entries of `split_ratios` must sum to 1.")
    class_to_split_sizes = get_class_split_sizes(class_sizes, split_ratios, class_to_idx)

    indices = list(range(len(dataset)))
    if shuffle:
        random.seed(42)
        random.shuffle(indices)

    # Allocate datapoints to splits.

    # The following maps are used to keep track, for each class, of which split is currently being
    # filled and to determine when it is full and we should switch to the next split.
    #   - class_to_current_split: maps each class label to the split being currently filled
    #   - class_to_current_size_in_current_split: maps each class to its current size in the
    #           split that is currenty getting filled

    processed_filenames = set()
    class_to_current_split = {class_to_idx[name]: 0 for name in class_names}
    class_to_current_size_in_current_split = {class_to_idx[name]: 0 for name in class_names}
    current_split_sizes = [0] * len(split_ratios)  # contains the current size of each split

    for index in tqdm.tqdm(indices, desc=dataset_path, total=len(dataset)):
        filename, cls = dataset.samples[index]
        try:
            with Image.open(filename) as f:
                # Just check if we can read the image.
                f.load()
        except OSError as e:
            logging.exception(e)
            logging.error(f"Image {filename} seems corrupted. Skipping...")
            continue


        # Construct a unique key from the filename.
        relative_filename = os.path.splitext(os.path.relpath(filename, dataset.root))[0]

        # Assert this filename has not been used so far.
        assert relative_filename not in processed_filenames, "Detected duplicate filename"
        processed_filenames.add(relative_filename)

        split_idx = class_to_current_split[cls]
        writers[split_idx].write(
            {
                "__key__": f"{current_split_sizes[split_idx]:07d}",
                "jpg": read_image(filename),
                "cls": cls,
                "_filename": relative_filename,
            }
        )
        current_split_sizes[split_idx] += 1
        class_to_current_size_in_current_split[cls] += 1

        # check if current split is full for class `cls` and switch to next one if yes
        if class_to_current_size_in_current_split[cls] == class_to_split_sizes[cls][split_idx]:
            class_to_current_split[cls] += 1
            class_to_current_size_in_current_split[cls] = 0

    for writer in writers:
        writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
