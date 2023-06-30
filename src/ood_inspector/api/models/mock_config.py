import dataclasses
from typing import Optional, Tuple

from ood_inspector.api.models import base_config


@dataclasses.dataclass
class MockModelConfig(base_config.ModelConfig):
    _target_: str = "ood_inspector.models.mock.MockModel"
    # Use ImageNet pretraining config by default
    pretraining_input_size: Optional[Tuple[int, int, int]] = (3, 224, 224)
    pretraining_input_mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    pretraining_input_std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
