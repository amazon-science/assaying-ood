import itertools
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, IterableDataset

from ood_inspector.datasets.transformation_stack import TransformationStack

logger = logging.getLogger(__name__)


class InspectorDataset(IterableDataset):
    """InspectorDataset separates transformations from the underlying dataset (untransformed)."""

    def __init__(
        self,
        dataset: Dataset,
        number_of_classes_per_attribute: Union[int, Dict[str, int]],
        default_attribute: str = "label_",
        transformations: Optional[TransformationStack] = None,
        input_size: Optional[Tuple[int, int, int]] = None,
        input_mean: Optional[Tuple[float, float, float]] = None,
        input_std: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Args:
            dataset (torch.utils.data.Dataset): A dataset returning a data dict with an image and
                one or multiple attributes.
            number_of_classes_per_attribute (Union[int, Dict[str, int]]): A dictionary containing
                the number of classes for each attribute. If the dataset has only one label, this
                parameter can be a single `int` value denoting the number of classes of the default
                attribute.
            default_attribute (str, optional): The default name of the target attribute used in
                experiments. The dataset must return a dict containing this key. Defaults to
                "label_".
            transformations (Optional[ood_inspector.datasets.TransformationStack], optional): An
                optional stack of transformations to be applied on each sample. If None, the samples
                are returned untransformed. Defaults to None.
            input_size (Optional[Tuple[int, int, int]], optional): The size of the images in the
                format (number_of_channels, height, width). Defaults to None.
            input_mean (Optional[Tuple[float, float, float]], optional): The mean of the dataset
                across all dimensions. Defaults to None.
            input_std (Optional[Tuple[float, float, float]], optional): The standard deviation of
                the dataset across all dimensions. Defaults to None.

        Example:
            ::

                >>> inspector_dataset = InspectorDataset(
                ...     dataset,
                ...     number_of_classes_per_attribute={"age": 100, "gender": 2},
                ...     default_attribute="age",
                ...     transformations=TransformationStack(
                ...         transforms.Compose(
                ...             [
                ...                 transforms.ToTensor(),
                ...                 transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ...             ]
                ...         )
                ...     ),
                ... )

                >>> next(iter(dataset))
                {'image': <PIL.Image.Image ...>, 'age': 26, 'gender': 0}

                >>> next(iter(inspector_dataset))
                {'image': tensor([[[2.2489, 2....2.6400]]]), 'age': 26, 'gender': 0}
        """

        if isinstance(number_of_classes_per_attribute, int):
            number_of_classes_per_attribute = {default_attribute: number_of_classes_per_attribute}

        if default_attribute not in number_of_classes_per_attribute:
            raise ValueError(
                f"Number of classes not provided for the default attribute '{default_attribute}'"
            )

        self.dataset = dataset
        self.number_of_classes_per_attribute = number_of_classes_per_attribute
        self.default_attribute = default_attribute
        self.transformations = transformations
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std

    def __iter__(self):
        for sample in self.dataset:
            yield self.transformations(sample) if self.transformations else sample

    def __len__(self):
        return len(self.dataset)

    @property
    def normalization(self):
        return self.transformations.normalization if self.transformations else None

    @property
    def attributes(self):
        return list(self.number_of_classes_per_attribute.keys())


class ChainedDataset(InspectorDataset):
    def __init__(self, datasets: List[InspectorDataset]) -> None:
        if not (
            ChainedDataset._check_uniqueness(datasets, "mean")
            and ChainedDataset._check_uniqueness(datasets, "std")
        ):
            raise ValueError("Not all normalization means or stds match within chain.")

        self._normalization = datasets[0].normalization

        # Verify that all datasets contain the same attributes
        for dataset in datasets[1:]:
            assert (
                dataset.number_of_classes_per_attribute
                == datasets[0].number_of_classes_per_attribute
            )

        chained_dataset = torch.utils.data.ChainDataset(datasets)

        super().__init__(
            dataset=chained_dataset,
            number_of_classes_per_attribute=datasets[0].number_of_classes_per_attribute,
            default_attribute=datasets[0].default_attribute,
            transformations=None,
        )

    @property
    def normalization(self):
        return self._normalization

    @staticmethod
    def _check_uniqueness(datasets, attribute):
        """Gets list of dataset attribute and checks them for quasi-uniqueness"""
        list_of_tensors = [getattr(ds.normalization, attribute) for ds in datasets]
        # sometimes the parameters are tensors, sometimes not
        list_of_tensors = [t if torch.is_tensor(t) else torch.tensor(t) for t in list_of_tensors]
        quantities_are_close = [
            torch.isclose(tensor_i, tensor_j).tolist()
            for tensor_i in list_of_tensors
            for tensor_j in list_of_tensors
        ]
        # depending on the number of channels we have to flatten the list
        if isinstance(quantities_are_close[0], list):
            quantities_are_close = list(itertools.chain.from_iterable(quantities_are_close))
        return all(quantities_are_close)
