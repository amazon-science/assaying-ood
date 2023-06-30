import logging
import os
import random
from itertools import islice
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
import tqdm
import webdataset
import wilds  # Not in requirements because needed only once -> `pip install wilds`.

logging.basicConfig(level=logging.INFO)


def get_dataset(root_dir, dataset_name, subset_name):
    dataset = wilds.get_dataset(
        dataset_name, unlabeled=False, root_dir=root_dir, split_scheme="official"
    )
    assert subset_name in dataset.split_dict, str((subset_name, dataset.split_dict.keys()))
    split = dataset.get_subset(subset_name)
    return split


def get_class_sizes(dataset: Sequence, indices: List[int] = None) -> Dict[str, int]:
    """Returns a map from a class name (an int) to the size of the class."""
    class_sizes = dict()
    # for _, cls, _ in tqdm.tqdm(dataset, desc="Counting class members", total=len(dataset)):
    indices = indices or range(len(dataset))
    for index in tqdm.tqdm(indices, desc="Counting class members", total=len(dataset)):
        _, cls, _ = dataset[index]
        cls = cls.to("cpu").item()
        assert isinstance(cls, int), type(cls)
        if cls in class_sizes:
            class_sizes[cls] += 1
        else:
            class_sizes[cls] = 1
    if hasattr(dataset, "n_classes") and len(class_sizes) != dataset.n_classes:
        logging.warning(
            f"Dataset's n_classes attribute = {dataset.n_classes} "
            f"but found {len(class_sizes)} classes in dataset."
        )
    return class_sizes


def get_class_split_sizes(
    class_sizes: Dict[int, int], split_ratios: Tuple[int]
) -> Dict[int, Tuple[int]]:
    """Returns a map from a class label to a tuple with the per-split data class sizes.

    The tuples all contain number_of_splits integers. The class labels are computed using the class
    the keys of class_sizes.
    """
    splitted_class_sizes = {
        cls: [int(split * size) for split in split_ratios] for cls, size in class_sizes.items()
    }
    relative_rounding_errors = {
        cls: [split * size - int(split * size) / (int(split * size) + 1) for split in split_ratios]
        for cls, size in class_sizes.items()
    }
    decreasing_error_indices = {
        cls: np.argsort(-np.array(errors), axis=-1)
        for cls, errors in relative_rounding_errors.items()
    }

    # For each class, distribute the remaining datapoints among splits one by one, starting with the
    # splits that have the highest relative rounding error
    for cls, decreasing_idx in decreasing_error_indices.items():
        for i in range(class_sizes[cls] - sum(splitted_class_sizes[cls])):
            splitted_class_sizes[cls][decreasing_idx[i]] += 1
    output = {cls: tuple(sizes) for cls, sizes in splitted_class_sizes.items()}
    return output


def write_webdataset(
    dataset: Sequence,
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
    label. Note that the dataset is loaded twice: a first time to get the the number of datapoints
    per class, and a second time to actually generate the webdataset.
    """

    if sum(split_ratios) != 1.0:
        raise ValueError("Entries of `split_ratios` must sum to 1.")

    indices = list(range(len(dataset)))
    cls_list = list()
    if shuffle:
        random.seed(42)
        random.shuffle(indices)

    metadata_keys = dataset.metadata_fields
    class_sizes = get_class_sizes(dataset, indices)
    classes = class_sizes.keys()
    class_to_split_sizes = get_class_split_sizes(class_sizes, split_ratios)

    # Allocate datapoints to splits.

    # The following maps are used to keep track, for each class, of which split is currently being
    # filled and to determine when it is full and we should switch to the next split.
    #   - class_to_current_split: maps each class label to the split being currently filled
    #   - class_to_current_size_in_current_split: maps each class to its current size in the
    #           split that is currenty getting filled

    class_to_current_split = {cls: 0 for cls in classes}
    class_to_current_size_in_current_split = {cls: 0 for cls in classes}
    current_split_sizes = [0] * len(split_ratios)  # Contains current size of each split.
    split_indices = [[] for _ in range(len(split_ratios))]  # list of indices in each split.

    description = "Webdataset creation"
    for index in tqdm.tqdm(indices, desc=description, total=len(dataset)):
        input, cls, metadata = dataset[index]
        cls = cls.to("cpu").item()
        cls_list.append(cls)
        assert isinstance(cls, int), type(cls)
        sample = {
            "__key__": f"{index:07d}",
            "png": input,
            "cls": cls,
        }
        for key, value in zip(metadata_keys, metadata):
            if key != "y":  # y = class label. Already in sample, so ignore.
                # webdataset treats keys that start with underscore as metadata.
                # Metadata must be a string.
                sample["_" + key] = str(value)
        split_idx = class_to_current_split[cls]
        writers[split_idx].write(sample)

        current_split_sizes[split_idx] += 1
        class_to_current_size_in_current_split[cls] += 1
        split_indices[split_idx].append(index)

        # Check if current split is full for class `cls` and switch to next one if yes.
        if class_to_current_size_in_current_split[cls] == class_to_split_sizes[cls][split_idx]:
            class_to_current_split[cls] += 1
            class_to_current_size_in_current_split[cls] = 0

    for writer in writers:
        writer.close()

    number_of_classes = dataset.n_classes if hasattr(dataset, "n_classes") else len(set(cls_list))
    split_sizes = tuple([current_split_sizes[i] for i in range(len(current_split_sizes))])
    dataset_size = sum(split_sizes)
    if dataset_size != len(dataset):
        logging.warning(
            f"Reported dataset size  = {len(dataset)} "
            f"but found {dataset_size} datapoints during webdataset creation"
        )

    return number_of_classes, split_sizes, tuple(split_indices)


def create_links_in_csv(
    dataset_name: str,
    s3_output_dir: str,
    name_tag: str,
    number_of_classes: int,
    split_names: Tuple[str],
    split_sizes: Tuple[int],
    shard_folders: Tuple[str],  # Path to shards on efs drive.
):
    """Create text for csv and append it to s3_webdatasets.csv."""
    for split_name, split_size, shard_folder in zip(split_names, split_sizes, shard_folders):
        shard_names = [name for name in os.listdir(shard_folder) if name.endswith(".tar")]
        max_shard_index = len(shard_names) - 1
        if max_shard_index == 0:
            shards = f"{s3_output_dir}/{split_name}/shard-000000.tar"
        else:
            shards = (
                f"{s3_output_dir}/{split_name}/shard-{'{'}000000..{max_shard_index:06d}{'}'}.tar"
            )

        text_for_csv = (
            f'"{name_tag}-{dataset_name}-{split_name}",'
            f'"{shards}",{split_size},"[(\'label_\', {number_of_classes})]"'
        )
        logging.info(text_for_csv)
        with open("s3_webdatasets.csv", "a") as file:
            file.write(text_for_csv + "\n")


def check_default__images(shard_folders, split_indices, dataset, base_pattern):
    """Check that first few images and labels in original and webdataset dataset are identical."""
    to_tensor = torchvision.transforms.ToTensor()
    for shard_folder, indices in zip(shard_folders, split_indices):
        uri = os.path.join(shard_folder, base_pattern % 0)
        wdataset = webdataset.WebDataset(uri).decode("pil").to_tuple("png", "cls")
        for (img1, cls1), index in zip(islice(wdataset, 0, 5), indices[:5]):
            # Assumes that wdataset and indices have at least 5 elements.
            img2, cls2, _ = dataset[index]
            img1 = to_tensor(img1)
            img2 = to_tensor(img2)
            if not torch.all(img1 == img2).item() or cls1 != cls2:
                raise ValueError(
                    f"Saved data does not match original data in {shard_folder}\n"
                    f"Input maxdiff: {torch.max(torch.abs(img1-img2)).item()}\n"
                    f"cls1: {cls1}, cls2: {cls2}"
                )
    logging.info(
        "Passed sanity check: in every split, first few webdataset images match the original ones."
    )


def main(
    dataset_name: str,  # A wilds dataset name.
    subset_name: str,  # An official wilds dataset split.
    root_dir: str,  # Path to folder with all wilds datasets.
    output_dir: str,  # Path to output folder, e.g. ~/wilds/dataset_name/custom_splits.
    s3_output_dir: str,  # Shards will be in "{s3_output_dir}/{split_name}/".
    name_tag: str,  # Prefix for name tag in csv file, e.g., S3Wilds.
    split_ratios: Tuple[float],
    split_names: Tuple[str],
    maxcount: int = 10000,
    maxsize: int = int(3e8),
    shuffle: bool = True,
):
    logging.info(f"Processing {dataset_name}/{subset_name}")
    base_pattern: str = "shard-%06d.tar"
    dataset = get_dataset(root_dir, dataset_name, subset_name)

    if len(split_names) == 1 and len(split_ratios) > 1:
        split_names = [f"split-{i}" for i in range(len(split_ratios))]
    assert len(set(split_names)) == len(split_names) == len(split_ratios)

    # TODO(pgehler): support non-sharded webdataset.TarWriter
    def writer(pattern):
        return webdataset.ShardWriter(pattern, maxsize=maxsize, maxcount=maxcount, keep_meta=True)

    writers = list()
    shard_folders = list()
    for split_name in split_names:
        shard_folder = os.path.join(output_dir, split_name)
        os.makedirs(shard_folder)
        writers.append(writer(os.path.join(shard_folder, base_pattern)))
        shard_folders.append(shard_folder)

    number_of_classes, split_sizes, split_indices = write_webdataset(
        dataset, tuple(writers), shuffle, split_ratios
    )

    create_links_in_csv(
        dataset_name,
        s3_output_dir,
        name_tag,
        number_of_classes,
        split_names,
        split_sizes,
        shard_folders,
    )

    check_default__images(shard_folders, split_indices, dataset, base_pattern)


if __name__ == "__main__":
    configs_camelyon17 = {
        "root_dir": "/efs/wilds/",
        "name_tag": "S3Wilds",
        "dataset_name": "camelyon17",
        "subset_name": "id_val",
        "output_dir": "/efs/webdata/wilds/camelyon17/custom_splits/",
        "s3_output_dir": "s3://inspector-data/sharded/wilds/camelyon17/custom_splits",
        "split_ratios": (0.5, 0.5),
        "split_names": ("id_val-subset-test", "id_val-subset-val"),
    }

    configs_rxrx1 = {
        "root_dir": "/efs/wilds/",
        "name_tag": "S3Wilds",
        "dataset_name": "rxrx1",
        "subset_name": "id_test",
        "output_dir": "/efs/webdata/wilds/rxrx1/custom_splits/",
        "s3_output_dir": "s3://inspector-data/sharded/wilds/rxrx1/custom_splits",
        "split_ratios": (0.5, 0.5),
        "split_names": ("id_test-subset-test", "id_test-subset-val"),
    }

    main(**configs_camelyon17)
    main(**configs_rxrx1)
