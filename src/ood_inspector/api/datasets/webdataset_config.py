import ast
import collections
import csv
import dataclasses
import os
from typing import Dict, List, Optional

import hydra.core.config_store as hydra_config_store
import omegaconf

from ood_inspector.api.datasets import datasets_config
from ood_inspector.datasets import dataset as dataset_utils


@dataclasses.dataclass
class WebDatasetConfig(datasets_config.DatasetConfig):
    _target_: str = "ood_inspector.datasets.webdataset.get_webdataset"
    uri_expression: str = omegaconf.MISSING
    number_of_classes_per_attribute: Optional[Dict[str, int]] = None
    number_of_datapoints: Optional[int] = None


# TODO (zietld): Add and use explicit dataset name.
def extract_datasetname_by_uri(uri):
    return "_".join(uri[5:].split("/")[2:-1])


omegaconf.OmegaConf.register_new_resolver("extract_datasetname_by_uri", extract_datasetname_by_uri)


@dataclasses.dataclass
class FewshotSubsampledDatasetConfig(datasets_config.DatasetConfig):
    _target_: str = "ood_inspector.datasets.webdataset.get_fewshot_subsampled_dataset"
    dataset: datasets_config.DatasetConfig = omegaconf.MISSING
    maxcount: int = 10000
    maxsize: int = int(3e8)
    number_datapoints_per_class: int = 10
    target_attribute: Optional[str] = omegaconf.MISSING  # "default_"
    number_of_classes_per_attribute: Dict[str, int] = omegaconf.MISSING
    force_create: bool = False
    s3_cache_folder: str = omegaconf.SI(
        "s3://inspector-data/sharded/subsample_cache/"
        + "${extract_datasetname_by_uri:${.dataset.uri_expression}}_"
        + "fewshot_${.number_datapoints_per_class}"
    )


S3DataInfo = collections.namedtuple(
    "S3DataInfo",
    [
        "name",
        "uri_expression",
        "number_of_datapoints",
        "number_of_classes_per_attribute",
    ],
)


_S3_WEBDATASETS_FAIRNESS = os.path.join(os.path.dirname(__file__), "s3_fairness_webdatasets.csv")
_S3_WEBDATASETS = os.path.join(os.path.dirname(__file__), "s3_webdatasets.csv")


def _s3_webdatasets(definitions_file: str) -> List[S3DataInfo]:
    datasets = []
    with open(definitions_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for (
            name,
            uri_expression,
            number_of_datapoints,
            number_of_classes_per_attribute,
        ) in reader:
            number_of_datapoints = ast.literal_eval(number_of_datapoints)
            number_of_classes_per_attribute = dict(
                ast.literal_eval(number_of_classes_per_attribute)
            )
            datasets.append(
                S3DataInfo(
                    name,
                    uri_expression,
                    number_of_datapoints,
                    number_of_classes_per_attribute,
                )
            )
    return datasets


def register_datasets(
    group: str,
    as_dict: bool,
    transformations: Optional[dataset_utils.TransformationStack] = None,
) -> None:
    config_store = hydra_config_store.ConfigStore.instance()
    transformations = transformations or datasets_config.TransformationStackConfig()

    for dataset in _s3_webdatasets(_S3_WEBDATASETS) + _s3_webdatasets(_S3_WEBDATASETS_FAIRNESS):
        inspector_dataset = datasets_config.InspectorDatasetConfig(
            dataset=WebDatasetConfig(
                uri_expression=dataset.uri_expression,
                number_of_datapoints=dataset.number_of_datapoints,
                number_of_classes_per_attribute=dataset.number_of_classes_per_attribute,
            ),
            number_of_classes_per_attribute=dataset.number_of_classes_per_attribute,
            default_attribute=list(dataset.number_of_classes_per_attribute.keys())[0],
            transformations=transformations,
        )

        if as_dict:
            # Store config in form of a dictionary
            config_store.store(
                group=group,
                name=dataset.name,
                node={dataset.name: inspector_dataset},
            )
        else:
            config_store.store(
                group=group,
                name=dataset.name,
                node=inspector_dataset,
            )


def register_dataset_presets() -> None:
    config_store = hydra_config_store.ConfigStore.instance()

    for dataset in _s3_webdatasets(_S3_WEBDATASETS) + _s3_webdatasets(_S3_WEBDATASETS_FAIRNESS):
        config_store.store(
            group="presets/datasets",
            name=dataset.name,
            node=WebDatasetConfig(
                uri_expression=dataset.uri_expression,
                number_of_datapoints=dataset.number_of_datapoints,
                number_of_classes_per_attribute=dataset.number_of_classes_per_attribute,
            ),
        )


def register_fewshot_datasets(
    group: str,
    number_datapoints_per_class: int,
    transformations: Optional[dataset_utils.TransformationStack] = None,
) -> None:
    config_store = hydra_config_store.ConfigStore.instance()
    transformations = transformations or datasets_config.TransformationStackConfig()

    for dataset in _s3_webdatasets(_S3_WEBDATASETS):
        config_store.store(
            group=group,
            name=f"{dataset.name}-fewshot-{number_datapoints_per_class}",
            node=datasets_config.InspectorDatasetConfig(
                dataset=FewshotSubsampledDatasetConfig(
                    dataset=WebDatasetConfig(
                        uri_expression=dataset.uri_expression,
                        number_of_datapoints=dataset.number_of_datapoints,
                        number_of_classes_per_attribute=dataset.number_of_classes_per_attribute,
                    ),
                    target_attribute=list(dataset.number_of_classes_per_attribute.keys())[0],
                    number_of_classes_per_attribute=dataset.number_of_classes_per_attribute,
                    number_datapoints_per_class=number_datapoints_per_class,
                ),
                transformations=transformations,
            ),
        )
