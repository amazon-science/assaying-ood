"""Ensure all hydra configs are registered."""
# Ignore checks here, we import these modules in order to ensure the hydra config store is populated
# but don't access them in this file.
# flake8: noqa

import ood_inspector.api.adaptation_config
import ood_inspector.api.attacks_config
import ood_inspector.api.augmentation_config
import ood_inspector.api.corruption_config
import ood_inspector.api.datasets
import ood_inspector.api.evaluations_config
import ood_inspector.api.inspector_config
import ood_inspector.api.models
import ood_inspector.api.torch_core_config
import ood_inspector.api.transform_config
