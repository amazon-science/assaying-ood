"""Inspector models"""
import enum
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

import ood_inspector.utils as utils
from ood_inspector.models import inspector_base

TimmModelName = enum.Enum(
    "TimmModelName",
    # This call lists both those models with pretrained weights and those without.
    timm.list_models(),
)


class TimmModel(inspector_base.InspectorModel):
    def __init__(
        self,
        model: nn.Module,
        pretraining_input_size: Optional[Tuple[int, int, int]] = None,
        pretraining_input_mean: Optional[Tuple[float, float, float]] = None,
        pretraining_input_std: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__(
            model, pretraining_input_size, pretraining_input_mean, pretraining_input_std
        )
        # Setup feature extraction.
        classifier_layers = self._get_classifier_layers()
        self._n_classifier_layers = len(classifier_layers)
        # TODO(cjsg): Improve feature_tracker when len(classifier_layers) > 0.
        # Currently, if len(classifier_layers), then only the first layer will be used in the
        # self.feature_tracker, which itself is used to compute n_features.
        self.feature_tracker = utils.track_input_of_layer(classifier_layers[0])[1]
        # Setup model properties.
        self._n_classes = self.model.default_cfg["num_classes"]
        # Needs to be determined using a forward pass in `setup()`.
        self._n_features = None

    @property
    def is_normalized(self) -> bool:
        return False

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def n_features(self) -> int:
        return self._n_features

    def _get_classifier_layers(self):
        """Returns a tuple containing the layer(s) used for classification by the timm model.

        Timm gives us some additional information about the classifier layer. This function parses
        the path to the classifier layer and returns the layer object.
        """
        classifier_paths = self.model.default_cfg["classifier"]
        if isinstance(classifier_paths, str):
            classifier_paths = (classifier_paths,)
        nodes = list()
        for classifier_path in classifier_paths:
            classifier_path = classifier_path.split(".")
            cur_node = self.model
            for fragment in classifier_path:
                cur_node = getattr(cur_node, fragment)
            nodes.append(cur_node)
        return tuple(nodes)

    def setup(self, inputs: torch.Tensor) -> None:
        self.model(inputs)
        self._n_features = self.feature_tracker.values[0].shape[-1]
        self._setup = True

    # TODO(pgehler): re-enable feature computation. Not all evaluations require setup.
    # @_ensure_setup
    def forward(self, inputs: torch.Tensor) -> inspector_base.InspectorModelOutput:
        logits = self.model(inputs)
        if isinstance(logits, tuple):  # Architecture has multiple classification heads.
            logits = torch.mean(torch.stack(logits, dim=0), dim=0)
        # features = self.feature_tracker.values[0]  # Not true if multiple classification heads.
        return inspector_base.InspectorModelOutput(logits=logits, features=None)

    def set_classification_head(self, number_of_classes: int) -> None:
        device = self.device
        self.model.reset_classifier(number_of_classes, global_pool="avg")
        self._n_classes = number_of_classes
        self.model.to(device)


def load_timm_model(
    name: enum.Enum,
    pretrained: bool,
    device: str,
    pretraining_input_size: Optional[Tuple[int, int, int]] = None,
    pretraining_input_mean: Optional[Tuple[float, float, float]] = None,
    pretraining_input_std: Optional[Tuple[float, float, float]] = None,
    **kwargs
) -> TimmModel:
    """Wrapper function to create a timm model.

    Needed in order to convert from AnyTimmModel to string for the function
    `timm.create_model`.
    """
    model = timm.create_model(name.name, pretrained=pretrained, **kwargs)
    return TimmModel(
        model.to(torch.device(device)),
        pretraining_input_size,
        pretraining_input_mean,
        pretraining_input_std,
    )
