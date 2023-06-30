"""Tests for Augmentation Utils."""
from typing import Dict
from unittest import TestCase

import numpy as np
import PIL
import pytest

try:
    from torchvision.transforms import transforms as tv_transforms

    import ood_inspector.augmentation as augmentations
    from ood_inspector.datasets import IMAGENET_MEAN, IMAGENET_STD
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)

RGB_IMAGENET_MEAN = tuple([round(255 * x) for x in IMAGENET_MEAN])


def white_image():
    return PIL.Image.new("RGB", (10, 20), (255, 255, 255))


def black_image():
    return PIL.Image.new("RGB", (10, 20), (0, 0, 0))


def asymmetric_image():
    return PIL.Image.fromarray(np.reshape(np.arange(10 * 20 * 3), (10, 20, 3)).astype("uint8"))


def get_torchvision_default_transform(use_normalization=False):
    transform = tv_transforms.Compose(
        [
            tv_transforms.RandomResizedCrop(224),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
        ]
    )
    if use_normalization:
        transform = tv_transforms.Compose(
            [
                transform,
                tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transform


@pytest.mark.parametrize(
    "augmenter,parameters",
    [
        (augmentations.BlurImageAugmenter, {}),
        (augmentations.ColorJitterImageAugmenter, {}),
        (augmentations.EncodingQualityImageAugmenter, {}),
        (augmentations.GrayScaleImageAugmenter, {}),
        (augmentations.HorizontalFlipImageAugmenter, {}),
        (augmentations.IdentityImageAugmenter, {}),
        (augmentations.PerspectiveImageAugmenter, {}),
        (augmentations.RandomNoiseImageAugmenter, {}),
        (augmentations.RandomPixelationImageAugmenter, {}),
        (augmentations.SharpenImageAugmenter, {}),
        (augmentations.AutoMixImageAugmenter, {}),
        (augmentations.AutoMixImageAugmenter, {"config_str": "rand"}),
        (augmentations.AutoMixImageAugmenter, {"config_str": "original"}),
        (augmentations.AutoMixImageAugmenter, {"config_str": "augmix-b1"}),
    ],
)
def test_shapes_image_augmenter(augmenter: augmentations.ImageAugmenter, parameters: Dict):
    augmentation_function = augmenter(**parameters)
    augmented_image = augmentation_function(asymmetric_image())
    TestCase().assertEqual(np.shape(augmented_image), np.shape(asymmetric_image()))


@pytest.mark.parametrize(
    "augmenter",
    [
        augmentations.BlurImageAugmenter,
        augmentations.ColorJitterImageAugmenter,
        augmentations.EncodingQualityImageAugmenter,
        augmentations.GrayScaleImageAugmenter,
        augmentations.HorizontalFlipImageAugmenter,
        augmentations.PerspectiveImageAugmenter,
        augmentations.RandomPixelationImageAugmenter,
        augmentations.RandomNoiseImageAugmenter,
        augmentations.SharpenImageAugmenter,
    ],
)
def test_image_augmenter_defaults(augmenter):
    """Checks that default parameters change a random image."""
    augmentation_function = augmenter()
    augmented_image = augmentation_function(asymmetric_image())

    is_equal = (np.array(augmented_image) == np.array(asymmetric_image())).all()
    TestCase().assertFalse(is_equal)


@pytest.mark.parametrize(
    "augmenter,image,parameters,expected",
    [
        (augmentations.BlurImageAugmenter, asymmetric_image(), {"radius": 2.0}, False),
        (augmentations.BlurImageAugmenter, black_image(), {"radius": 2.0}, True),
        (augmentations.ColorJitterImageAugmenter, white_image(), {"brightness_factor": 0.0}, False),
        (augmentations.ColorJitterImageAugmenter, black_image(), {"brightness_factor": 0.0}, True),
        (augmentations.ColorJitterImageAugmenter, white_image(), {"contrast_factor": 0.0}, True),
        (augmentations.ColorJitterImageAugmenter, black_image(), {"contrast_factor": 0.0}, True),
        (augmentations.ColorJitterImageAugmenter, white_image(), {"saturation_factor": 0.0}, True),
        (augmentations.ColorJitterImageAugmenter, white_image(), {"saturation_factor": 0.0}, True),
        (augmentations.EncodingQualityImageAugmenter, asymmetric_image(), {"quality": 20}, False),
        (augmentations.GrayScaleImageAugmenter, black_image(), {}, True),
        (augmentations.HorizontalFlipImageAugmenter, white_image(), {}, True),
        (augmentations.IdentityImageAugmenter, white_image(), {}, True),
        (augmentations.RandomNoiseImageAugmenter, asymmetric_image(), {"var": 0.0}, True),
        (augmentations.RandomNoiseImageAugmenter, asymmetric_image(), {"var": 1.0}, False),
        (
            augmentations.RandomPixelationImageAugmenter,
            asymmetric_image(),
            {"min_ratio": 1.0},
            True,
        ),
        (augmentations.SharpenImageAugmenter, asymmetric_image(), {"factor": 1.0}, True),
    ],
)
def test_identity_image_augmenter(augmenter, image, parameters, expected):
    augmentation_function = augmenter(**parameters)
    augmented_image = augmentation_function(image)

    is_equal = (np.array(augmented_image) == np.array(image)).all()
    TestCase().assertEqual(is_equal, expected)


@pytest.mark.parametrize(
    "augmenter,parameters,expected",
    [
        (augmentations.BlurImageAugmenter, {}, {"radius": 1.0, "p": 1.0}),
        (augmentations.BlurImageAugmenter, {"radius": 2.0}, {"radius": 2.0, "p": 1.0}),
        (
            augmentations.ColorJitterImageAugmenter,
            {},
            {"brightness_factor": 1.1, "saturation_factor": 1.1, "contrast_factor": 1.1, "p": 1.0},
        ),
        (
            augmentations.ColorJitterImageAugmenter,
            {"brightness_factor": 0.0},
            {"brightness_factor": 0.0, "saturation_factor": 1.1, "contrast_factor": 1.1, "p": 1.0},
        ),
        (augmentations.EncodingQualityImageAugmenter, {}, {"quality": 10, "p": 1.0}),
        (augmentations.GrayScaleImageAugmenter, {}, {"mode": "luminosity", "p": 1.0}),
        (augmentations.HorizontalFlipImageAugmenter, {}, {"p": 1.0}),
        (augmentations.IdentityImageAugmenter, {}, {}),
        (
            augmentations.PerspectiveImageAugmenter,
            {},
            {"sigma": 50.0, "dx": 0.0, "dy": 0.0, "seed": 42, "p": 1.0},
        ),
        (augmentations.RandomNoiseImageAugmenter, {}, {"var": 0.01, "p": 1.0}),
        (augmentations.RandomNoiseImageAugmenter, {"var": 1.0}, {"var": 1.0, "p": 1.0}),
        (
            augmentations.RandomPixelationImageAugmenter,
            {},
            {"min_ratio": 0.1, "max_ratio": 1.0, "p": 1.0},
        ),
        (augmentations.SharpenImageAugmenter, {}, {"factor": 8.0, "p": 1.0}),
        (
            augmentations.AutoMixImageAugmenter,
            {
                "config_str": "rand-m9",
                "img_mean": RGB_IMAGENET_MEAN,
            },
            {
                "config_str": "rand-m9",
                "kwargs": {"img_mean": RGB_IMAGENET_MEAN},
            },
        ),
        (
            augmentations.RandomPixelationImageAugmenter,
            {},
            {"min_ratio": 0.1, "max_ratio": 1.0, "p": 1.0},
        ),
        (augmentations.SharpenImageAugmenter, {}, {"factor": 8.0, "p": 1.0}),
    ],
)
def test_parameters_image_augmenter(augmenter, parameters, expected):
    augmentation_function = augmenter(**parameters)
    TestCase().assertDictEqual(expected, augmentation_function.parameters)


@pytest.mark.parametrize(
    "augmenter,parameters",
    [
        (augmentations.BlurImageAugmenter, {}),
        (augmentations.RandomNoiseImageAugmenter, {}),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        get_torchvision_default_transform(use_normalization=False),
        get_torchvision_default_transform(use_normalization=True),
    ],
)
def test_augmenter_compose_with_transform(augmenter, parameters, transform):
    image = asymmetric_image()
    augmentation_function = augmenter(**parameters)
    combined = augmentation_function.compose_with_transform(transform)
    assert combined(image) is not None


@pytest.mark.parametrize(
    "deterministic_augmenters",
    [
        (
            augmentations.GrayScaleImageAugmenter(p=1.0),
            augmentations.HorizontalFlipImageAugmenter(p=1.0),
        )
    ],
)
def test_compose_image_augmenter(deterministic_augmenters):
    image = asymmetric_image()
    composition_augmenter = augmentations.ComposeImageAugmenter(deterministic_augmenters)
    new_image = image
    for augmenter in deterministic_augmenters:
        new_image = augmenter(new_image)
    composed_image = composition_augmenter(image)
    assert (np.array(composed_image) == np.array(new_image)).all()


@pytest.mark.parametrize(
    "config_str,kwargs",
    [
        ("rand-m9-mstd0.5-inc1", dict()),
        ("original-mstd0.5", dict()),
        ("augmix-m5-w4-d2-b1", dict()),
        ("augmix-m5-w4-d2-b1", {"interpolation": PIL.Image.BICUBIC}),
    ],
)
def test_automix_image_augmenter_variants(config_str, kwargs):
    image = asymmetric_image()
    width, height = image.size
    augmenter = augmentations.AutoMixImageAugmenter(config_str, **kwargs)
    assert augmenter(image) is not None
