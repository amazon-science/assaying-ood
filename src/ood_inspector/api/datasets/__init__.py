import hydra.core.config_store as hydra_config_store

from . import webdataset_config
from .datasets_config import (
    AdaptationTransformationStackConfig,
    DatasetConfig,
    EvaluationTransformationStackConfig,
    InspectorDatasetConfig,
    TransformationStackConfig,
)
from .webdataset_config import FewshotSubsampledDatasetConfig, WebDatasetConfig, _s3_webdatasets

config_store = hydra_config_store.ConfigStore.instance()
# Register schemas.
config_store.store(
    group="schemas/datasets",
    name="InspectorDataset",
    node=InspectorDatasetConfig,
)

config_store.store(
    group="schemas/datasets",
    name="FewshotSubsampledDataset",
    node=FewshotSubsampledDatasetConfig,
)

config_store.store(
    group="schemas/datasets",
    name="TransformationStack",
    node=TransformationStackConfig,
)

config_store.store(
    group="schemas/datasets",
    name="AdaptationTransformationStack",
    node=AdaptationTransformationStackConfig,
)

config_store.store(
    group="schemas/datasets",
    name="EvaluationTransformationStack",
    node=EvaluationTransformationStackConfig,
)


transformations = EvaluationTransformationStackConfig()
for group in ["datasets", "corruption.datasets"]:
    webdataset_config.register_datasets(group, as_dict=True, transformations=transformations)
# Register presets for advanced configuration in yaml files.
webdataset_config.register_dataset_presets()


# Register datasets for adaptation.
transformations = AdaptationTransformationStackConfig()
webdataset_config.register_datasets(
    "adaptation.dataset", as_dict=False, transformations=transformations
)


for datapoints_per_class in [10, 100]:
    webdataset_config.register_fewshot_datasets(
        "adaptation.dataset", datapoints_per_class, transformations=transformations
    )
