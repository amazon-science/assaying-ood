"""Functions for data augmentation."""
from typing import Any, Callable, Dict, Optional, Sequence

import augly.image.transforms as augly_transforms
from timm.data import auto_augment as timm_aa
from torchvision.transforms import transforms as tv_transforms

from ood_inspector.datasets import IMAGENET_MEAN


class ImageAugmenter:
    """Generic augmentation class."""

    transform: Callable
    parameters: Dict

    def __init__(self, transform_function: Callable, parameters: Optional[Dict] = None):
        """Initialize augmentation."""
        self._transform_function = transform_function
        self._parameters = parameters or dict()
        super().__init__()

    def __call__(self, image):
        """Applies the augmentation to the image and returns the augmented image."""
        return self._transform_function(image)

    def compose_with_transform(self, transform: Any):
        """Appends augmentation to given transfrom and returns torchvision transform object.

        Args:
            transform: The output type is expected to be a torch.Tensor. E.g., use
                tv_tranforms.ToTensor() in the transforms list.

        Returns:
            Torchvision transformation with output type torch.Tensor.
        """
        combined = tv_transforms.Compose(
            [
                transform,
                tv_transforms.ToPILImage(),
                self,
                tv_transforms.ToTensor(),
            ]
        )
        return combined

    @property
    def parameters(self):
        """Gets the parameters of the augmentation.

        Returns:
            Dictionary containing the augmentation parameters. If the augmentation has no
            parameters, the dictionary is empty.
        """
        return self._parameters

    def __repr__(self):
        if isinstance(self._transform_function, augly_transforms.BaseTransform):
            # Native __repr__() of augly transforms is awful, so we make it prettier
            parameters_str = ", ".join(
                [f"{name}={value}" for name, value in self.parameters.items()]
            )
            return f"augly_transforms.{type(self._transform_function).__name__}({parameters_str})"
        else:
            return self._transform_function.__repr__()

    def __str__(self):
        if isinstance(self._transform_function, augly_transforms.BaseTransform):
            return self.__repr__()
        else:
            return self._transform_function.__str__()


class IdentityImageAugmenter(ImageAugmenter):
    def __init__(self, parameters: Optional[Dict] = None):
        """Identity augmentation."""
        del parameters
        super().__init__(transform_function=lambda x: x)


class HorizontalFlipImageAugmenter(ImageAugmenter):
    def __init__(self, p: float = 1.0):
        """Flips the image along the horizontal axis with probability p."""
        parameters = {"p": p}
        super().__init__(transform_function=augly_transforms.HFlip(p), parameters=parameters)


class ColorJitterImageAugmenter(ImageAugmenter):
    def __init__(
        self,
        brightness_factor: float = 1.1,
        contrast_factor: float = 1.1,
        saturation_factor: float = 1.1,
        p: float = 1.0,
    ):
        """Applies color jitter augmentation with probability p.

        Args:
            brightness_factor: Float, a factor of one leaves the brightness unchanged. A factor
                greater than one brightens the image and a factor smaller than one darkens it.
            contrast_factor: Float, a factor of one leaves the contrast unchanged. A factor greater
                than one adds contrast and a factor smaller than one removes contrast.
            brightness_factor: Float, a factor of one leaves the saturation unchanged. A factor
                greater than one increases the saturation and smaller than one decreases it.
            p: Float, probability of applying the transform.
        """
        parameters_dict = {
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "saturation_factor": saturation_factor,
            "p": p,
        }

        super().__init__(
            transform_function=augly_transforms.ColorJitter(**parameters_dict),
            parameters=parameters_dict,
        )


class BlurImageAugmenter(ImageAugmenter):
    def __init__(self, radius: float = 1.0, p: float = 1.0):
        """Applies blur augmentation with a specific radius with probability p.

        Args:
            radius: Float determining how blurry the image will be. Needs to be greater than 0.
            p: Float, probability of applying the transform.

        """
        parameters_dict = {"radius": radius, "p": p}

        super().__init__(
            transform_function=augly_transforms.Blur(**parameters_dict), parameters=parameters_dict
        )


class RandomNoiseImageAugmenter(ImageAugmenter):
    def __init__(self, var: float = 0.01, p: float = 1.0):
        """Adds Gaussian noise with zero mean and a specific variance.

        Args:
            var: Float determining the variance of the noise Gaussian. Needs to be greater than 0.
            p: Float, probability of applying the transform.

        """
        parameters_dict = {"var": var, "p": p}

        super().__init__(
            transform_function=augly_transforms.RandomNoise(**parameters_dict),
            parameters=parameters_dict,
        )


class GrayScaleImageAugmenter(ImageAugmenter):
    def __init__(self, mode: str = "luminosity", p: float = 1.0):
        """Turn image to grayscale.

        Args:
            mode: the type of greyscale conversion to perform; two options are supported
                ("luminosity" and "average")
            p: Float, probability of applying the transform.
        """
        parameters_dict = {"mode": mode, "p": p}
        super().__init__(
            transform_function=augly_transforms.Grayscale(**parameters_dict),
            parameters=parameters_dict,
        )


class PerspectiveImageAugmenter(ImageAugmenter):
    def __init__(
        self, sigma: float = 50.0, dx: float = 0.0, dy: float = 0.0, seed: int = 42, p: float = 1.0
    ):
        """Perspective transformation, image looks like a photo from another device.

        Args:
            sigma: Standard deviation of the distribution of destination coordinates. Larger sigma
                implies stronger augmentation.
            dx: Change in the x coordinate for the perspective transform. Mean of the distribution
                of destination coordinates on the x axis.
            dy: Change in the y coordinate for the perspective transform. Mean of the distribution
                of destination coordinates on the y axis.
            p: Float, probability of applying the transform.
        """
        parameters_dict = {"sigma": sigma, "dx": dx, "dy": dy, "seed": seed, "p": p}

        super().__init__(
            transform_function=augly_transforms.PerspectiveTransform(**parameters_dict),
            parameters=parameters_dict,
        )


class RandomPixelationImageAugmenter(ImageAugmenter):
    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 1.0, p: float = 1.0):
        """Random pixelation of the image.

        Args:
            min_ratio: Lowest value in the range of pixelation. Smaller values result in more
                pixelated image. Values greater or equal than one do not change the image.
            max_ratio: Highest value in the range of pixelation. Smaller values result in more
                pixelated image. Values greater or equal than one do not change the image.
            p: Float, probability of applying the transform.
        """
        parameters_dict = {"min_ratio": min_ratio, "max_ratio": max_ratio, "p": p}

        super().__init__(
            transform_function=augly_transforms.RandomPixelization(**parameters_dict),
            parameters=parameters_dict,
        )


class SharpenImageAugmenter(ImageAugmenter):
    def __init__(self, factor: float = 8.0, p: float = 1.0):
        """Sharpening augmentation.

        Args:
            factor: A value lower than one blurs the image, one leaves it unchanged, and grater than
                one sharpens it.
            p: Float, probability of applying the transform.
        """
        parameters_dict = {"factor": factor, "p": p}

        super().__init__(
            transform_function=augly_transforms.Sharpen(**parameters_dict),
            parameters=parameters_dict,
        )


class EncodingQualityImageAugmenter(ImageAugmenter):
    def __init__(self, quality: int = 10, p: float = 1.0):
        """Sharpening augmentation.

        Args:
            quality: JPEG encoding quality between 0 and 100. A factor of 100 is not the identity.
            p: Float, probability of applying the transform.
        """
        parameters_dict = {"quality": quality, "p": p}

        super().__init__(
            transform_function=augly_transforms.EncodingQuality(**parameters_dict),
            parameters=parameters_dict,
        )


class ComposeImageAugmenter(ImageAugmenter):
    def __init__(self, image_augmenters: Sequence[ImageAugmenter]):
        super().__init__(transform_function=tv_transforms.Compose(image_augmenters))


class AutoMixImageAugmenter(ImageAugmenter):
    def __init__(
        self,
        config_str: str = "rand-m9-mstd0.5-inc1",  # Default from facebookresearch/deit.
        **kwargs: Any,
    ):
        """
        Creates an auto-augment, random-augment or aug-mix augmenter from timm.

        The augmentation type (auto-augment, random-augment or augmix) and its main parameters get
        defined in config_str. Note that this implementation of random-augment does not
        include the `cutout` (i.e., random erasing) transform contrary to the original paper,
        because timm handles it as a separate transform.

        Inputs:
            config_str: String that defines the augmentation type and its parameters.
                Consists of multiple sections separated by dashes ('-'). The first section defines
                the specific variant of auto augment ('rand', 'augmix', 'original', ...). The
                remaining sections, not order sepecific determine, define parameters for the chosen
                augmentation.  See doc of timm's `auto_augment_transform`,
                `random_augment_transform` and `augment_and_mix_transform` functions for more
                details.

                Examples:
                    - ``'original-mstd0.5'`` gives AutoAugment with original policy, magnitude_std
                      0.5

                    - ``'rand-m9-n3-mstd0.5'`` gives RandAugment with magnitude 9, num_layers 3,
                      magnitude_std 0.5

                    - ``'rand-mstd1-w0'`` results in magnitude_std 1.0, weights 0, default magnitude
                      of 10 and num_layers 2

                    - ``'augmix-m5-w4-d2-b1'`` yields AugMix with severity 5, chain width 4, chain
                      depth 2, using an accelerated mixing method called blending

            kwargs: Optional hyper parameters to be passed to timm's auto-augmentation functions
                such as 'img_mean', 'interpolation' or 'translate_pct'. See timm's doc and code.
        """
        assert isinstance(config_str, str)

        if "img_mean" not in kwargs:
            # Default value used by timm to fill blank pixels arising e.g. from image translations.
            kwargs["img_mean"] = tuple([min(255, round(255 * x)) for x in IMAGENET_MEAN])

        if config_str.startswith("rand"):
            transform_function = timm_aa.rand_augment_transform(config_str, kwargs)
        elif config_str.startswith("augmix"):
            # With asymmetric images, default augmix fails, due to a bug in timm; use "-b1" option.
            transform_function = timm_aa.augment_and_mix_transform(config_str, kwargs)
        else:
            transform_function = timm_aa.auto_augment_transform(config_str, kwargs)

        parameters = dict()
        parameters["config_str"] = config_str
        parameters["kwargs"] = kwargs
        super().__init__(transform_function=transform_function, parameters=parameters)
