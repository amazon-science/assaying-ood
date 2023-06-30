API configs
===========

Inspector Dataset Config
------------------------
.. autoclass:: ood_inspector.api.datasets.DatasetConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ood_inspector.api.datasets.InspectorDatasetConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ood_inspector.api.datasets.TransformationStackConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ood_inspector.api.datasets.AdaptationTransformationStackConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ood_inspector.api.datasets.EvaluationTransformationStackConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ood_inspector.api.datasets.WebDatasetConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: ood_inspector.api.datasets.FewshotSubsampledDatasetConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autofunction:: ood_inspector.api.datasets._s3_webdatasets

Inspector Model Config
----------------------
.. autoclass:: ood_inspector.api.models.base_config.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Evaluation Metrics Config
-------------------------
.. automodule:: ood_inspector.api.evaluations_config
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :noindex:

Corruptions Config
------------------
.. automodule:: ood_inspector.api.corruption_config
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Transformations Config
----------------------
.. automodule:: ood_inspector.api.transform_config
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: use_default_if_none
   :noindex:

Augmentations Config
--------------------
.. automodule:: ood_inspector.api.augmentation_config
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: register_augmenters
   :noindex:

Adaptation Config
-----------------
.. automodule:: ood_inspector.api.adaptation_config
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: ood_inspector.api.torch_core_config
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: convert_fractional_to_integer_steps
   :noindex:
