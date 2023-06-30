import dataclasses
from typing import Any, Optional, Sequence

import hydra.core.config_store as hydra_config_store
import omegaconf


@dataclasses.dataclass
class AugmenterConfig:
    pass


@dataclasses.dataclass
class BlurImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.BlurImageAugmenter"


@dataclasses.dataclass
class ColorJitterImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.ColorJitterImageAugmenter"
    brightness_factor: Optional[float] = 1.1
    contrast_factor: Optional[float] = 1.1
    saturation_factor: Optional[float] = 1.1


@dataclasses.dataclass
class EncodingQualityImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.EncodingQualityImageAugmenter"


@dataclasses.dataclass
class GrayScaleImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.GrayScaleImageAugmenter"
    p: float = 1.0
    mode: str = "luminosity"


@dataclasses.dataclass
class HorizontalFlipImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.HorizontalFlipImageAugmenter"


@dataclasses.dataclass
class IdentityImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.IdentityImageAugmenter"


@dataclasses.dataclass
class PerspectiveImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.PerspectiveImageAugmenter"


@dataclasses.dataclass
class RandomNoiseImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.RandomNoiseImageAugmenter"


@dataclasses.dataclass
class RandomPixelationImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.RandomPixelationImageAugmenter"


@dataclasses.dataclass
class SharpenImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.SharpenImageAugmenter"


@dataclasses.dataclass
class ComposeImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.ComposeImageAugmenter"
    image_augmenters: Sequence[AugmenterConfig] = omegaconf.MISSING


@dataclasses.dataclass
class ColormixImageAugmenterConfig(ComposeImageAugmenterConfig):
    image_augmenters: Any = (
        ColorJitterImageAugmenterConfig(
            brightness_factor=0.3, contrast_factor=0.3, saturation_factor=0.3
        ),
        GrayScaleImageAugmenterConfig(p=0.1),
    )


@dataclasses.dataclass
class AutoMixImageAugmenterConfig(AugmenterConfig):
    _target_: str = "ood_inspector.augmentation.AutoMixImageAugmenter"
    config_str: str = omegaconf.MISSING


def register_augmenters(group):
    # TODO(cjsg): unify naming conventions for "name" (underscores or WordCapitalization)
    config_store = hydra_config_store.ConfigStore.instance()
    config_store.store(group=group, name="blur", node=BlurImageAugmenterConfig)
    config_store.store(group=group, name="color_jitter", node=ColorJitterImageAugmenterConfig)
    config_store.store(
        group=group, name="encoding_quality", node=EncodingQualityImageAugmenterConfig
    )
    config_store.store(group=group, name="gray_scale", node=GrayScaleImageAugmenterConfig)
    config_store.store(group=group, name="horizontal_flip", node=HorizontalFlipImageAugmenterConfig)
    config_store.store(group=group, name="identity", node=IdentityImageAugmenterConfig)
    config_store.store(group=group, name="perspective", node=PerspectiveImageAugmenterConfig)
    config_store.store(group=group, name="random_noise", node=RandomNoiseImageAugmenterConfig)
    config_store.store(
        group=group, name="random_pixelation", node=RandomPixelationImageAugmenterConfig
    )
    config_store.store(group=group, name="sharpen", node=SharpenImageAugmenterConfig)
    config_store.store(group=group, name="random_color", node=ColormixImageAugmenterConfig)
    config_store.store(
        group=group,
        name="automix",
        node=AutoMixImageAugmenterConfig,
    )
    config_store.store(
        group=group,
        name="random_augment",
        node=AutoMixImageAugmenterConfig(config_str="rand-m9-mstd0.5-inc1"),  # From facebook deit.
    )
    config_store.store(
        group=group,
        name="auto_augment",
        node=AutoMixImageAugmenterConfig(config_str="original-mstd0.5"),  # Example from timm doc.
    )
    config_store.store(
        group=group,
        name="augmix",
        node=AutoMixImageAugmenterConfig(config_str="augmix-m5-w4-d2-b1"),  # Example from timm doc.
    )


register_augmenters("adaptation.dataset.transformations.augmenter")
