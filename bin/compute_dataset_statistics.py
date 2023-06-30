"""Compute dataset statistics."""
import collections
import dataclasses
import json
import logging
import traceback
from typing import Any, Dict, List

import hydra
import hydra.core.config_store as hydra_config_store
import tqdm

import ood_inspector.api  # noqa: F401
from ood_inspector.api.datasets.datasets_config import DatasetConfig
from ood_inspector.api.datasets.webdataset_config import (
    _S3_WEBDATASETS,
    WebDatasetConfig,
    _s3_webdatasets,
)
from ood_inspector.api.models.mock_config import MockModelConfig

LOGGER = logging.getLogger(__name__)


class Statistic:
    """A statistic that should be computed on a dataset."""

    def reset(self):
        pass

    def update(self, x, y):
        pass

    def compute(self):
        pass


class InstancesPerClass(Statistic):
    def __init__(self):
        self.reset()

    def reset(self):
        self._class_count = collections.defaultdict(lambda: 0)

    def update(self, x, y):
        self._class_count[y] += 1

    def compute(self):
        # Although dicts are unordered in older versions of python, new versions keep the ordering.
        # This makes the output a bit easier to parse.
        return {clss: self._class_count[clss] for clss in sorted(self._class_count.keys())}


class NumberOfInstances(Statistic):
    def __init__(self):
        self.reset()

    def reset(self):
        self._instance_count = 0

    def update(self, x, y):
        self._instance_count += 1

    def compute(self):
        return self._instance_count


class NumberOfClasses(Statistic):
    def __init__(self):
        self.reset()

    def reset(self):
        self._classes = set()

    def update(self, x, y):
        self._classes.add(y)

    def compute(self):
        return len(self._classes)


_STATISTICS = {
    "instances_per_class": InstancesPerClass(),
    "number_of_instances": NumberOfInstances(),
    "number_of_classes": NumberOfClasses(),
}


@dataclasses.dataclass
class ComputeStatisticsConfig:
    output_path: str
    model: MockModelConfig = MockModelConfig()
    datasets: Dict[str, DatasetConfig] = dataclasses.field(default_factory=dict)
    defaults: List[Any] = dataclasses.field(
        default_factory=lambda: [{"datasets": "all_webdatasets"}, "_self_"]
    )


config_store = hydra_config_store.ConfigStore.instance()
# Register additional convenience settings.
config_store.store(
    name="all_webdatasets",
    group="datasets",
    node={
        dataset.name: WebDatasetConfig(
            uri_expression=dataset.uri_expression,
            number_of_datapoints=dataset.number_of_datapoints,
            number_of_classes=dataset.number_of_classes,
        )
        for dataset in _s3_webdatasets(_S3_WEBDATASETS)
    },
)

config_store.store(name="compute_statistics", node=ComputeStatisticsConfig)


@hydra.main(config_name="compute_statistics", config_path=None)
def compute_statistics(config: ComputeStatisticsConfig):
    """Compute summary statistics of datasets."""
    output_path = config.output_path
    datasets = hydra.utils.instantiate(config.datasets)
    output: Dict[str, Any] = collections.defaultdict(dict)

    datasets_iterator = tqdm.tqdm(datasets.items(), total=len(datasets), desc="Processing datasets")
    try:
        for name, dataset in datasets_iterator:
            for statistic in _STATISTICS.values():
                statistic.reset()

            dataset_iterator = tqdm.tqdm(dataset, total=len(dataset), desc=name, position=1)
            for x, y in dataset_iterator:
                for statistic in _STATISTICS.values():
                    statistic.update(x, y)

            for statistic_name, statistic in _STATISTICS.items():
                output[name][statistic_name] = statistic.compute()
    except Exception:
        LOGGER.error("Caught Exception during processing!")
        traceback.print_exc()
        LOGGER.warning("Writing incomplete results...")

    row_names, rows = zip(*output.items())
    with open(output_path, "w") as f:
        json.dump(dict(output), f, indent=2)


if __name__ == "__main__":
    compute_statistics()
