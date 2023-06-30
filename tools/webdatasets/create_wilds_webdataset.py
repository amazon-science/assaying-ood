import logging
import os
import random
from itertools import islice

import torch
import torchvision
import tqdm
import webdataset
import wilds  # Not in requirements because needed only once -> `pip install wilds`.

ROOT_DIR = "/efs/wilds/"
OUTPUT_DIR = "/efs/webdata/wilds/"
NAME_TAG = "S3Wilds"
S3LOCATION = "s3://inspector-data/sharded/wilds"
# Ignoring "poverty" bc regression (not classification) problem with 7-channels (not 3)
DATASET_NAMES = ("fmow", "iwildcam", "camelyon17", "rxrx1")
SHUFFLE = True
MAXCOUNT = 10000
MAXSIZE = int(3e8)


def writer(pattern):
    return webdataset.ShardWriter(pattern, maxsize=MAXSIZE, maxcount=MAXCOUNT, keep_meta=True)


base_pattern: str = "shard-%06d.tar"

for dataset_name in DATASET_NAMES:
    logging.info(f"Processing {dataset_name}")
    dataset = wilds.get_dataset(
        dataset_name, unlabeled=False, root_dir=ROOT_DIR, split_scheme="official"
    )

    for split_name in dataset.split_dict:
        logging.info(f"Processing {dataset_name}/{split_name}")
        relative_shards_folder = os.path.join(dataset_name, "official_splits", split_name)
        shards_folder = os.path.join(OUTPUT_DIR, relative_shards_folder)
        os.makedirs(shards_folder, exist_ok=True)
        split = dataset.get_subset(split_name)
        metadata_keys = split.metadata_fields
        indices = list(range(len(split)))
        cls_list = list()
        if SHUFFLE:
            random.seed(42)
            random.shuffle(indices)

        with writer(os.path.join(shards_folder, base_pattern)) as sink:
            description = f"{dataset_name}/{split_name}"
            for counter, index in enumerate(tqdm.tqdm(indices, desc=description, total=len(split))):
                input, cls, metadata = split[index]
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
                sink.write(sample)

        # Create text for csv
        shard_names = [name for name in os.listdir(shards_folder) if name.endswith(".tar")]
        max_shard_index = len(shard_names) - 1
        if max_shard_index == 0:
            shards = f"{S3LOCATION}/{relative_shards_folder}/shard-000000.tar"
        else:
            shards = (
                f"{S3LOCATION}/{relative_shards_folder}/"
                f"shard-{'{'}000000..{max_shard_index:06d}{'}'}.tar"
            )
        number_of_classes = len(set(cls_list))
        if number_of_classes != split.n_classes:
            logging.warning(
                f"Dataset's n_classes attribute = {split.n_classes} "
                f"but found {number_of_classes} classes during webdataset creation"
            )
            number_of_classes = split.n_classes
        dataset_size = counter + 1
        if dataset_size != len(split):
            logging.warning(
                f"Reported dataset size  = {len(split)} "
                f"but found {dataset_size} datapoints during webdataset creation"
            )

        # Add "ood_" prefix to "val" and "test" splits to avoid confusion with the inspector naming
        # convention, where val and test sets are typically in-distribution (=id). Wilds dataset
        # typically also have official id_val and id_test sets, but since they were not generated
        # with our own dataset splitting algo (which ensures class balancing), we keep the "id_"
        # prefix to avoid any confusions later.
        s3_split_name = f"ood_{split_name}" if split_name in {"val", "test"} else split_name

        text_for_csv = (
            f'"{NAME_TAG}-{dataset_name}-{s3_split_name}",'
            f'"{shards}",{dataset_size},"[(\'label_\', {number_of_classes})]"'
        )
        logging.info(text_for_csv)
        with open("s3_webdatasets.csv", "a") as file:
            file.write(text_for_csv + "\n")

        # Check that first few images match
        to_tensor = torchvision.transforms.ToTensor()
        uri = os.path.join(shards_folder, "shard-000000.tar")
        wdataset = webdataset.WebDataset(uri).decode("pil").to_tuple("png", "cls")
        for (img1, cls1), index in zip(islice(wdataset, 0, 5), indices[:5]):
            img2, cls2, _ = split[index]
            img1 = to_tensor(img1)
            img2 = to_tensor(img2)
            if not torch.all(img1 == img2).item() or cls1 != cls2:
                raise ValueError(
                    f"Saved data does not match original data in {dataset_name}/{split_name}\n"
                    f"Input maxdiff: {torch.max(torch.abs(img1-img2)).item()}\n"
                    f"cls1: {cls1}, cls2: {cls2}"
                )
