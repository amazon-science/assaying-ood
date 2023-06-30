import abc
import enum
import functools
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import foolbox
import torch
import tqdm
from torch import nn
from torchvision import transforms

from ood_inspector import attacks
from ood_inspector.models import inspector_base

logger = logging.getLogger(__name__)


def _raise_if_not_normalized(outputs: torch.Tensor):
    probabilities_sum = torch.sum(outputs, dim=-1)
    if not torch.all(torch.isclose(probabilities_sum, torch.ones_like(probabilities_sum))):
        raise ValueError(
            "The given probabilites have to sum to 1 in the last dimension. To activate automatic "
            " normalization of the outputs, set in the config `model.is_normalized=False`."
        )


class Evaluation(metaclass=abc.ABCMeta):
    """Interface for evaluations."""

    def __init__(self, target_attribute: str = "default_"):
        self.target_attribute = target_attribute

    @abc.abstractmethod
    def setup(self, model: inspector_base.InspectorModel, normalization_transform: Any):
        """Setup estimator to be ready for update calls.

        This is in order to give the estimator access to additional variables such as the model and
        do setup steps prior to receiving data. Setup should also reset the internal state of the
        estimator in order to ensure updates are performed correctly on a clean state.

        The base class implementation does nothing.

        Args:
            model (Optional[inspector_base.InspectorModel]): Model being evaluated.
            normalization_transform: Normalization transformation used by the current dataset.
        """

    def update(
        self, inputs: torch.tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        """Update the internal state of the estimator to reflect additional data.

        Base class does nothing.

        Args:
            inputs (torch.tensor): Input to the model.
            outputs (torch.Tensor): Outputs of the model.
            all_labels (Dict[str, torch.Tensor]): All labels of the current input.

        """

    @abc.abstractmethod
    def score(self) -> Union[float, Mapping[str, Any]]:
        """Return the score from the estimator.

        Returns:
            Union[float, Mapping[str, Any]]:
        """

    @property
    @abc.abstractmethod
    def requires_data(self) -> bool:
        """Determines if the evaluation requires access to data."""

    @property
    def apply_on_corrupted_datasets(self) -> bool:
        """Determines if the evaluation should be applied on corrupted datasets."""
        return True


def adapt_batch_size(update):
    @functools.wraps(update)
    def batch_size_adaptation_wrapper(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        """A wrapper to reduce an evaluation's batch size when facing a CUDA out of memory error.

        This decorator is meant to wrap the update function of Evaluation classes (such as
        AutoAttackEvaluation or AdversarialAccuracyEvaluation) and is agnostic to the inner workings
        of this update. When facing a CUDA out of memory error, it will halve the batch size and
        keep halving it until the error is gone or the batch size = 1. The final batch size is
        stored as a new `_internal_batch_size` attribute of the evaluation instance and will be used
        as the initial batch size at the next call to update. To perform the halving of the batch
        size, it chunks the actual input batch into two and passes each chunk in sequence to the
        update method.
        """
        if not hasattr(self, "_internal_batch_size"):
            # Batch size of the inputs that get passed to `update`. Gets halved if update faces a
            # CUDA out of memory error. Gets saved for all subsequent calls to `update`.
            self._internal_batch_size = len(inputs)
        index = 0
        while self._internal_batch_size > 1 and index < len(inputs):
            torch.cuda.empty_cache()
            try:
                subbatch_size = self._internal_batch_size
                inputs_subbatch = inputs[index : (index + subbatch_size)]
                all_labels_subbatch = {
                    k: l[index : (index + subbatch_size)] for k, l in all_labels.items()
                }
                outputs_subbatch = outputs[index : (index + subbatch_size)]
                update(self, inputs_subbatch, outputs_subbatch, all_labels_subbatch)
                index += subbatch_size
            except RuntimeError as e:
                error_message = str(e)
                if "CUDA out of memory" in error_message and self._internal_batch_size > 1:
                    self._internal_batch_size = int(self._internal_batch_size // 2)
                    logger.warning(
                        f"Got '{error_message}' while running the {update.__name__} method of "
                        f"{type(self).__name__}. Reducing batch size to "
                        f"{self._internal_batch_size} for retry."
                    )
                else:
                    raise e

    return batch_size_adaptation_wrapper


class FairnessEvaluation(Evaluation):
    def __init__(self, target_attribute: str = "default_", group_attribute: str = "default_"):
        Evaluation.__init__(self, target_attribute)
        self.group_attribute = group_attribute

    @abc.abstractmethod
    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
        number_of_classes_per_attribute: Dict,
    ):
        """Setup estimator to be ready for update calls.

        This is in order to give the estimator access to additional variables such as the model and
        do setup steps prior to receiving data. Setup should also reset the internal state of the
        estimator in order to ensure updates are performed correctly on a clean state.

        The base class implementation does nothing.

        Args:
            model (Optional[inspector_base.InspectorModel]): Model being evaluated.
            normalization_transform: Normalization transformation used by the current dataset.
            number_of_classes_per_attribute: Dict providing the number of classes per attribute.
        """


class AlwaysZero(Evaluation):
    """Dummy metric that always returns zero."""

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        pass

    def update(
        self, inputs: torch.tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        pass

    def score(self):
        return 0.0

    def __str__(self) -> str:
        return "Returns always 0."

    @property
    def requires_data(self):
        return False


class NumberOfParameters(Evaluation):
    """Number of model parameters."""

    number_of_parameters: int

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        self.number_of_parameters = model.number_of_parameters

    def score(self):
        return self.number_of_parameters

    def __str__(self):
        return "Number of model parameters"

    @property
    def requires_data(self):
        return False


class ClassificationAccuracy(nn.Module, Evaluation):
    """Top-k multi-class classification accuracy.

    A prediction is considered to be correct if the true label is among the largest k predicted
    class probabilities. For top_k = 1, this reduces to standard classification accuracy.
    """

    n_correct: torch.Tensor
    n_scored: torch.Tensor

    def __init__(self, target_attribute: str = "default_", top_k: int = 1) -> None:
        """
        Args:
            target_attribute: Attribute with respect to which we evaluate.
            top_k: How many positions of the top classes to consider.
        """
        nn.Module.__init__(self)
        Evaluation.__init__(self, target_attribute)
        self._top_k = top_k
        self._invalid_top_k = False

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        self._invalid_top_k = False
        try:
            self.n_correct.zero_()
            self.n_scored.zero_()
        except AttributeError:
            self.register_buffer("n_correct", torch.zeros((1,), dtype=torch.int64, device="cpu"))
            self.register_buffer("n_scored", torch.zeros((1,), dtype=torch.int64, device="cpu"))

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        del inputs
        labels = all_labels[self.target_attribute]
        if self._invalid_top_k:
            return

        if self._top_k == 1:
            predicted_class = outputs.argmax(-1)
            self.n_correct.add_((predicted_class == labels).long().sum())
        else:
            if outputs.dim() == 1:
                outputs = outputs[None, :]  # Add batch dimension.

            if outputs.shape[-1] < self._top_k:
                self._invalid_top_k = True
            else:
                top_k_labels = outputs.topk(self._top_k, -1).indices
                top_k_correct = labels[:, None].expand_as(top_k_labels) == top_k_labels
                self.n_correct.add_((torch.any(top_k_correct, -1)).long().sum())
        self.n_scored.add_(labels.numel())

    def score(self) -> float:
        if self._invalid_top_k:
            logger.warning(
                "Provided top_k was larger than number of classes in at least one batch."
            )
            return float("nan")

        return float(self.n_correct / self.n_scored)

    def __str__(self) -> str:
        return f"Top-{self._top_k} classification accuracy"

    @property
    def requires_data(self) -> bool:
        return True


class AutoAttackEvaluation(nn.Module, Evaluation):
    def __init__(
        self,
        attack_size: float,
        target_attribute: str = "default_",
        number_of_samples: Optional[int] = None,
        skip_corruption_datasets: bool = False,
        adjust_attack_to_number_of_classes: bool = False,
        **attack_parameters: Any,
    ):
        """Initializes an adversarial accuracy evaluation based on the AutoAttack

        The argument `attack_parameters` should contain all parameters to be passed to the
        AutoAttack constructor, plus possibly one key called 'custom'. See doc from
        `create_auto_attack` function in ood_inspector.attacks.

        References
        ----------
            * Original paper of the attack: https://arxiv.org/abs/2003.01690
            * Also used in RobustBench:
                * paper: https://arxiv.org/abs/2010.09670
                * website: https://robustbench.github.io/
        """
        nn.Module.__init__(self)
        Evaluation.__init__(self, target_attribute)
        attack_parameters["eps"] = attack_size
        self._attack_size = attack_size
        self._attack_parameters = attack_parameters
        self._number_of_samples = number_of_samples
        self._skip_corruption_datasets = skip_corruption_datasets
        self.adjust_attack_to_number_of_classes = adjust_attack_to_number_of_classes
        self._attack_is_adjusted = False

    def _unnormalize_images(self, image: torch.Tensor) -> torch.Tensor:
        mean = self._mean.view(-1, 1, 1)
        std = self._std.view(-1, 1, 1)
        return image * std + mean

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        try:
            self.n_correct.zero_()
            self.n_scored.zero_()
        except AttributeError:
            self.register_buffer("n_correct", torch.zeros((1,), dtype=torch.int64, device="cpu"))
            self.register_buffer("n_scored", torch.zeros((1,), dtype=torch.int64, device="cpu"))

        self.device = model.device
        mean, std = self._get_normalization_values(normalization_transform)
        self._mean = self._attack_parameters.pop("mean", mean).to(self.device)
        self._std = self._attack_parameters.pop("std", std).to(self.device)
        wrapped_model = inspector_base.ModelWrapperForLogits(model, mean=self._mean, std=self._std)
        wrapped_model = wrapped_model.to(self.device)
        self._auto_attack = attacks.create_auto_attack(
            wrapped_model, device=self.device, **self._attack_parameters
        )

    @adapt_batch_size
    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        if self.adjust_attack_to_number_of_classes and not self._attack_is_adjusted:
            attacks.adjust_autoattack_to_number_of_classes(outputs.shape[-1], self._auto_attack)
            self._attack_is_adjusted = True

        if self._number_of_samples and int(self.n_scored) > self._number_of_samples:
            # Skip update routine if number of processed inputs > number_of_samples.
            return

        inputs = inputs.to(self.device)
        labels = all_labels[self.target_attribute].to(self.device)
        del outputs
        inputs = self._unnormalize_images(inputs)
        with torch.enable_grad():
            _, adversarial_labels = self._auto_attack.run_standard_evaluation(
                inputs, labels, bs=len(inputs), return_labels=True
            )
        attack_success = (adversarial_labels != labels).to("cpu")
        self.n_correct.add_((~attack_success).long().sum(axis=0))
        self.n_scored.add_(attack_success.shape[0])

    def score(self) -> float:
        # Remove auto_attack in order to avoid serialization issues.
        del self._auto_attack
        return float(self.n_correct / self.n_scored)

    def __str__(self, key: str = "AutoAttack", parameter_dict: Optional[Dict] = None) -> str:
        parameter_dict = parameter_dict or self._attack_parameters
        output = [f"{key}:"]
        for key, value in self._attack_parameters.items():
            output.append(f"{key}: {str(value)}")
        return "\n    ".join(output)

    def _get_normalization_values(
        self, normalization_transform: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if normalization_transform is None:
            return torch.tensor([0.0]), torch.tensor([1.0])
        if not isinstance(normalization_transform, transforms.Normalize):
            raise ValueError(
                "transform or transform[-1] must be a torchvision.transforms.Normalize instance. "
                f" But provided transform of type {type(normalization_transform)}."
            )
        mean, std = normalization_transform.mean, normalization_transform.std
        mean = mean if type(mean) == torch.Tensor else torch.tensor(mean)
        std = std if type(std) == torch.Tensor else torch.tensor(std)
        return mean, std

    @property
    def requires_data(self) -> bool:
        return True

    @property
    def apply_on_corrupted_datasets(self) -> bool:
        return not self._skip_corruption_datasets


class AdversarialAccuracy(nn.Module, Evaluation):
    def __init__(
        self,
        attack: foolbox.attacks.base.Attack,
        attack_sizes: Sequence[float] = (0.0,),
        number_of_samples: Optional[int] = None,
        skip_corruption_datasets: bool = False,
        target_attribute: str = "default_",
    ):
        nn.Module.__init__(self)
        Evaluation.__init__(self, target_attribute)
        self._attack = attack
        self._attack_sizes = attack_sizes
        self._number_of_samples = number_of_samples
        self._skip_corruption_datasets = skip_corruption_datasets

    def _unnormalize_images(self, image: torch.Tensor) -> torch.Tensor:
        mean = self._mean.view(-1, 1, 1)
        std = self._std.view(-1, 1, 1)
        return image * std + mean

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        try:
            self.n_correct.zero_()
            self.n_scored.zero_()
        except AttributeError:
            self.register_buffer(
                "n_correct", torch.zeros((len(self._attack_sizes),), dtype=torch.long, device="cpu")
            )
            self.register_buffer("n_scored", torch.zeros((1,), dtype=torch.long, device="cpu"))

        self.device = model.device
        mean, std = self._get_normalization_values(normalization_transform)
        self._mean, self._std = mean.to(self.device), std.to(self.device)
        preprocessing = dict(
            mean=torch.atleast_1d(self._mean), std=torch.atleast_1d(self._std), axis=-3
        )

        model_with_logit_outputs = inspector_base.ModelWrapperForLogits(model)
        self._foolbox_model = foolbox.PyTorchModel(
            model_with_logit_outputs,
            bounds=(0.0, 1.0),
            preprocessing=preprocessing,
            device=model.device,
        )

    @adapt_batch_size
    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        if self._number_of_samples is not None and int(self.n_scored) > self._number_of_samples:
            # Skip update routine if number of processed inputs > number_of_samples.
            return

        inputs = inputs.to(self.device)
        labels = all_labels[self.target_attribute].to(self.device)
        del outputs
        # attack_success.shape = len(attack_sizes), batch_size
        inputs = self._unnormalize_images(inputs)
        with torch.enable_grad():
            _, _, attack_success = self._attack(
                self._foolbox_model, inputs, labels, epsilons=self._attack_sizes
            )
        attack_success = attack_success.to("cpu")
        self.n_correct.add_((~attack_success).long().sum(axis=1))
        self.n_scored.add_(attack_success.shape[1])

    def score(self) -> Mapping[str, List[float]]:
        """Returns robust accuracy (1st list) for each attack size (2nd list)."""
        # Remove model in order to avoid serialization issues.
        del self._foolbox_model
        return {
            "adversarial_accuracies": (self.n_correct / self.n_scored).tolist(),
            "attack_sizes": list(self._attack_sizes),
        }

    def __str__(self) -> str:
        return "Adversarial classification accuracy"

    def _get_normalization_values(
        self, normalization_transform: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if normalization_transform is None:
            return torch.tensor([0.0]), torch.tensor([1.0])
        if not isinstance(normalization_transform, transforms.Normalize):
            raise ValueError(
                "transform or transform[-1] must be a torchvision.transforms.Normalize instance. "
                f" But provided transform of type {type(normalization_transform)}."
            )
        mean, std = normalization_transform.mean, normalization_transform.std
        mean = mean if type(mean) == torch.Tensor else torch.tensor(mean)
        std = std if type(std) == torch.Tensor else torch.tensor(std)
        return mean, std

    @property
    def requires_data(self) -> bool:
        return True

    @property
    def apply_on_corrupted_datasets(self) -> bool:
        return not self._skip_corruption_datasets


class ReliabilityDiagram(nn.Module, Evaluation):
    """Reliability diagram which supports batched estimation."""

    def __init__(
        self,
        target_attribute: str = "default_",
        number_bins: int = 10,
    ) -> None:
        """Initializes reliability diagram.

        Args:
            number_bins (int): Number of bins to use.
        """
        nn.Module.__init__(self)
        Evaluation.__init__(self, target_attribute)
        self.number_bins = number_bins

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        try:
            self.confidence_bins.zero_()
            self.accuracy_bins.zero_()
            self.bin_counts.zero_()
        except AttributeError:
            self.register_buffer(
                "confidence_bins", torch.zeros(self.number_bins, dtype=torch.float32, device="cpu")
            )
            self.register_buffer(
                "accuracy_bins", torch.zeros(self.number_bins, dtype=torch.float32, device="cpu")
            )
            self.register_buffer(
                "bin_counts", torch.zeros(self.number_bins, dtype=torch.int64, device="cpu")
            )

        self._model_is_normalized = model.is_normalized

    def _maybe_normalize(self, outputs: torch.Tensor) -> torch.Tensor:
        if self._model_is_normalized:
            _raise_if_not_normalized(outputs)
            return outputs
        return torch.softmax(outputs, dim=-1)

    def _scalar_to_unit_histogram_bins(self, input_data: torch.tensor, number_bins: int):
        """Maps a scalar between 0 and 1 to one of number_bins."""
        # PyTorch doesn't have a function where we can lookup the bin of an individual value
        # (histc is not sufficient). The current approach is only applicable to uniform bins. This
        # might be extended if needed.
        bin_size = 1.0 / number_bins
        bins = torch.floor(torch.div(input_data, bin_size)).long()
        # When predicted value is perfect 1.0, the bin could overflow.
        bins.masked_fill_(bins == number_bins, number_bins - 1)
        return bins

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        """Updates internal state to reflect additional data.

        Args:
            inputs (torch.tensor): Input to the model.
            outputs (torch.Tensor): Outputs of the model.
            all_labels (Dict[str, torch.Tensor]): All labels of the current input.

        """
        outputs = self._maybe_normalize(outputs)
        outputs = outputs
        labels = all_labels[self.target_attribute]

        confidence, predicted_class = torch.max(outputs, dim=-1)
        correct_prediction = (labels == predicted_class).float()
        bins = self._scalar_to_unit_histogram_bins(confidence, self.number_bins)
        self.bin_counts.add_(torch.bincount(bins, minlength=self.number_bins))
        self.confidence_bins.index_add_(0, bins, confidence)
        self.accuracy_bins.index_add_(0, bins, correct_prediction)

    def score(self) -> Dict[str, torch.Tensor]:
        """Computes the reliability diagram based on previously provided data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                confidence_bins, accuracy_bins, number_of_samples_per_bin
        """
        # Prevent division by zero.
        confidence = torch.where(
            self.bin_counts != 0,
            self.confidence_bins / self.bin_counts,
            torch.full_like(self.confidence_bins, float("nan")),
        )
        accuracy = torch.where(
            self.bin_counts != 0,
            self.accuracy_bins / self.bin_counts,
            torch.full_like(self.accuracy_bins, float("nan")),
        )

        return {"confidence": confidence, "accuracy": accuracy, "count": self.bin_counts}

    @property
    def requires_data(self) -> bool:
        return True


class ExpectedCalibrationError(ReliabilityDiagram):
    """Expected calibration error."""

    def __init__(
        self,
        target_attribute: str = "default_",
        number_bins: int = 10,
    ):
        super().__init__(target_attribute, number_bins)

    def score(self) -> torch.Tensor:
        """Computes the expected calibration error on previously seen data.

        Returns:
            torch.Tensor: The expected calibration error.
        """
        reliability_diagram = super().score()
        confidence_bins, accuracy_bins, bin_counts = (
            reliability_diagram["confidence"],
            reliability_diagram["accuracy"],
            reliability_diagram["count"],
        )
        bin_weight = bin_counts / bin_counts.sum()
        bin_contributions = bin_weight * torch.abs(accuracy_bins - confidence_bins)
        bin_contributions = torch.where(
            torch.isnan(bin_contributions), torch.zeros_like(bin_contributions), bin_contributions
        )
        return float(bin_contributions.sum())

    def __str__(self) -> str:
        return "Expected calibration error"

    @property
    def requires_data(self) -> bool:
        return True


class CollectionItem(enum.Enum):
    INPUTS = enum.auto()
    LABELS = enum.auto()
    OUTPUTS = enum.auto()


class _CollectValues(Evaluation):
    """Collect values from forward passes of models."""

    collected_values: Dict[CollectionItem, List]

    def __init__(
        self,
        to_collect: List[CollectionItem],
        number_of_samples: Optional[int] = None,
        target_attribute: str = "default_",
    ):
        super().__init__(target_attribute)
        self.to_collect = to_collect
        self.number_of_samples = number_of_samples
        if self.number_of_samples is None:
            logger.warning(
                f"{self.__class__.__name__}: Number of samples is not defined. "
                "This runs the evaluation on the complete dataset and could lead to memory issues."
            )

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        self.collected_values = {}
        self.n_collected = 0
        for item in self.to_collect:
            self.collected_values[item] = []

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        labels = all_labels[self.target_attribute]
        if self.number_of_samples and (self.n_collected > self.number_of_samples):
            # Skip collection of more values if we reached number of samples.
            return
        # Collect values and store them on the cpu to avoid GPU memory issues.
        for item in self.to_collect:
            if item is CollectionItem.INPUTS:
                self.collected_values[item].append(inputs)
            elif item is CollectionItem.OUTPUTS:
                self.collected_values[item].append(outputs)
            elif item is CollectionItem.LABELS:
                self.collected_values[item].append(labels)
        self.n_collected += inputs.shape[0]

    def score(self):
        return None

    @property
    def requires_data(self) -> bool:
        return True


class _EnvironmentInference(_CollectValues):
    """Infer environments minimizing calibration.

    This was introduced in the paper "Environment Inference for Invariant Learning"
    https://arxiv.org/pdf/2010.07249.pdf. It maximizes the invariant risk minimization
    penalty (irm).

    Args:
        number_of_samples (int): Number of samples for environment inference. Default is dataset
          size.
        number_of_steps (int): Number of optimization steps.
    """

    def __init__(
        self,
        number_of_samples: Optional[int] = None,
        number_of_steps: int = 10000,
        target_attribute: str = "default_",
    ) -> None:
        super().__init__(
            [CollectionItem.OUTPUTS, CollectionItem.LABELS], number_of_samples, target_attribute
        )
        self.number_of_steps = number_of_steps

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        super().setup(model, normalization_transform)
        self._model_device = model.device

    @staticmethod
    def _gradient(loss, scale_parameter):
        return torch.autograd.grad(loss, [scale_parameter], create_graph=True)[0]

    def _irm_penalty(self, loss, scale_parameter):
        return torch.sum(torch.pow(self._gradient(loss, scale_parameter), 2))

    def score(self) -> Dict:
        """Partitions the data to maximize the invariant risk minimization penalty (irm).

        Returns:
            Tuple[torch.Tensor, torch.Tensor] containing the mask that assigns each test
            example to either environment and the model logits.
        """

        logits = torch.cat(self.collected_values[CollectionItem.OUTPUTS], axis=0)
        del self.collected_values[CollectionItem.OUTPUTS]
        labels = torch.cat(self.collected_values[CollectionItem.LABELS], axis=0)
        del self.collected_values[CollectionItem.LABELS]

        # Gradient computation is deactivated for evaluations, we need it here though.
        with torch.enable_grad():
            scale = torch.ones(1, device=self._model_device, requires_grad=True).squeeze()
            loss = nn.functional.cross_entropy(
                logits.to(self._model_device) * scale,
                labels.to(self._model_device),
                reduction="none",
            )

            assignment_weights = torch.randn(
                len(logits), device=self._model_device, requires_grad=True
            )
            optimizer = torch.optim.Adam([assignment_weights], lr=0.001)
            for _ in tqdm.tqdm(range(self.number_of_steps), desc="Inferring environments"):
                # Invariant Risk Minimization penalty (irm)
                loss_environment1 = (loss * assignment_weights.sigmoid()).mean()
                irm_environment1 = self._irm_penalty(loss_environment1, scale)

                loss_environment2 = (loss * (1.0 - assignment_weights.sigmoid())).mean()
                irm_environment2 = self._irm_penalty(loss_environment2, scale)

                irm_loss = -torch.stack([irm_environment1, irm_environment2]).mean()
                optimizer.zero_grad()
                irm_loss.backward(retain_graph=True)
                optimizer.step()

        return {
            "environment": (assignment_weights.sigmoid() > 0.5).detach().to("cpu"),
            "logits": logits,
        }


class DemographicParityInferredGroups(_EnvironmentInference):
    """Compute demographic parity with inferred environments.

    Adpatation of the unfairness metric in "On the Fairness of Disentangled Representations".
    https://arxiv.org/pdf/1905.13662.pdf which uses the total variation to extend demographic
    parity to non-binary classifiers.

    """

    def demographic_parity_with_groups(
        self, predictions: torch.Tensor, environment_mask: torch.Tensor, number_of_classes: int
    ):

        predictions_counts = torch.bincount(predictions, minlength=number_of_classes)

        predictions_first_environment = predictions[environment_mask]
        predictions_counts_first_env = torch.bincount(
            predictions_first_environment, minlength=number_of_classes
        )

        predictions_second_environment = predictions[~environment_mask]
        predictions_counts_second_env = torch.bincount(
            predictions_second_environment, minlength=number_of_classes
        )

        predictions_distribution = predictions_counts / predictions_counts.sum()

        distribution_first_environment = predictions_counts_first_env / max(
            predictions_counts_first_env.sum(), 1
        )

        distribution_second_environment = predictions_counts_second_env / max(
            predictions_counts_second_env.sum(), 1
        )

        total_variation_first_env = 0.5 * torch.sum(
            torch.abs(predictions_distribution - distribution_first_environment)
        )
        total_variation_second_env = 0.5 * torch.sum(
            torch.abs(predictions_distribution - distribution_second_environment)
        )

        total_predictions = predictions_counts_first_env.sum() + predictions_counts_second_env.sum()
        weight_first_env = predictions_counts_first_env.sum() / total_predictions
        weight_second_env = predictions_counts_second_env.sum() / total_predictions
        return (
            weight_first_env * total_variation_first_env
            + weight_second_env * total_variation_second_env
        ).item()

    def score(self):
        environment_inference = super().score()
        environment = environment_inference["environment"]
        predictions = environment_inference["logits"].argmax(-1).long()
        n_classes = environment_inference["logits"].shape[-1]
        del environment_inference

        return self.demographic_parity_with_groups(predictions, environment, n_classes)

    def __str__(self) -> str:
        return "Demographic parity inferred groups"


class NegativeLogLikelihood(nn.Module, Evaluation):
    """Multi-class negative log likelihood.

    If the predicted vector of probabilities is [p_1, ..., p_K] and the true class label is k,
    then the negative log likelihood is -log(p_k).
    """

    def __init__(
        self,
        target_attribute: str = "default_",
    ):
        super().__init__()
        Evaluation.__init__(self, target_attribute)

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
    ):
        try:
            self.loss_sum.zero_()
            self.n_score.zero_()
        except AttributeError:
            self.register_buffer("loss_sum", torch.zeros((1,), dtype=torch.float32, device="cpu"))
            self.register_buffer("n_scored", torch.zeros((1,), dtype=torch.int64, device="cpu"))
        self._model_is_normalized = model.is_normalized

    def _maybe_normalize_and_log(self, outputs: torch.Tensor) -> torch.Tensor:
        if self._model_is_normalized:
            _raise_if_not_normalized(outputs)
            return outputs.log()
        return torch.log_softmax(outputs, dim=-1)

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        del inputs
        labels = all_labels[self.target_attribute]
        # nll_loss expects log probabilites.
        outputs = self._maybe_normalize_and_log(outputs)
        neg_log_likelihood = torch.nn.functional.nll_loss(outputs, labels, reduction="sum")
        self.loss_sum.add_(neg_log_likelihood)
        self.n_scored.add_(labels.numel())

    def score(self) -> float:
        return float(self.loss_sum / self.n_scored)

    def __str__(self) -> str:
        return "Negative log likelihood"

    @property
    def requires_data(self) -> bool:
        return True


class ClassificationAccuracyPerGroup(nn.Module, FairnessEvaluation):
    """Top-k multi-class classification accuracy evaluated per group; groups are defined as
    datapoints that share a common value of the group attribute.

    A prediction is considered to be correct if the true label is among the largest k predicted
    class probabilities. For top_k = 1, this reduces to standard classification accuracy.
    """

    n_correct: torch.Tensor
    n_scored: torch.Tensor

    def __init__(
        self, target_attribute: str = "default_", group_attribute: str = "default_", top_k: int = 1
    ) -> None:
        """
        Args:
            target_attribute: Attribute with respect to which we evaluate.
            group_attribute: Attribute defining the groups.
            top_k: How many positions of the top classes to consider.
        """
        nn.Module.__init__(self)
        FairnessEvaluation.__init__(self, target_attribute, group_attribute)
        self._top_k = top_k
        self._invalid_top_k = False

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
        number_of_classes_per_attribute: Dict,
    ):
        self.number_of_classes_per_attribute = number_of_classes_per_attribute
        self.number_of_groups = self.number_of_classes_per_attribute[self.group_attribute]
        self._invalid_top_k = False
        try:
            self.n_correct.zero_()
            self.n_scored.zero_()
        except AttributeError:
            self.register_buffer(
                "n_correct", torch.zeros((self.number_of_groups,), dtype=torch.int64, device="cpu")
            )
            self.register_buffer(
                "n_scored", torch.zeros((self.number_of_groups,), dtype=torch.int64, device="cpu")
            )

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        del inputs
        labels = all_labels[self.target_attribute]
        groups = all_labels[self.group_attribute]
        if self._invalid_top_k:
            return

        if self._top_k == 1:
            predicted_class = outputs.argmax(-1)
            self.n_correct.add_(
                torch.tensor(
                    [
                        (predicted_class == labels)[groups == gr].long().sum()
                        for gr in range(self.number_of_groups)
                    ]
                )
            )
        else:
            if outputs.dim() == 1:
                outputs = outputs[None, :]  # Add batch dimension.

            if outputs.shape[-1] < self._top_k:
                self._invalid_top_k = True
            else:
                top_k_labels = outputs.topk(self._top_k, -1).indices
                top_k_correct = labels[:, None].expand_as(top_k_labels) == top_k_labels
                self.n_correct.add_(
                    torch.tensor(
                        [
                            (torch.any(top_k_correct, -1))[groups == gr].long().sum()
                            for gr in range(self.number_of_groups)
                        ]
                    )
                )
        self.n_scored.add_(
            torch.tensor([labels[groups == gr].numel() for gr in range(self.number_of_groups)])
        )

    def score(self) -> float:
        if self._invalid_top_k:
            logger.warning(
                "Provided top_k was larger than number of classes in at least one batch."
            )
            return float("nan")

        return {
            gr: float(self.n_correct[gr] / self.n_scored[gr]) for gr in range(self.number_of_groups)
        }

    def __str__(self) -> str:
        return f"Top-{self._top_k} classification accuracy per group"

    @property
    def requires_data(self) -> bool:
        return True


class OutputDistributionPerGroup(nn.Module, FairnessEvaluation):
    """Distribution of output response evaluated per group; this can be used to compute violations
    of demographic parity, for example; groups are defined as datapoints that share a common value
    of the group attribute.
    """

    n_correct: torch.Tensor
    n_scored: torch.Tensor

    def __init__(
        self, target_attribute: str = "default_", group_attribute: str = "default_", top_k: int = 1
    ) -> None:
        """
        Args:
            target_attribute: Attribute with respect to which we evaluate, i.e., the attribute
                              defining the output.
            group_attribute: Attribute defining the groups.
        """
        nn.Module.__init__(self)
        FairnessEvaluation.__init__(self, target_attribute, group_attribute)

    def setup(
        self,
        model: inspector_base.InspectorModel,
        normalization_transform: Any,
        number_of_classes_per_attribute: Dict,
    ):
        self.number_of_classes_per_attribute = number_of_classes_per_attribute
        self.number_of_groups = self.number_of_classes_per_attribute[self.group_attribute]
        self.number_of_output_classes = self.number_of_classes_per_attribute[self.target_attribute]
        try:
            self.n_scored.zero_()
        except AttributeError:
            self.register_buffer(
                "n_scored",
                torch.zeros(
                    (self.number_of_groups, self.number_of_output_classes),
                    dtype=torch.int64,
                    device="cpu",
                ),
            )

    def update(
        self, inputs: torch.Tensor, outputs: torch.Tensor, all_labels: Dict[str, torch.Tensor]
    ) -> None:
        del inputs
        groups = all_labels[self.group_attribute]
        predicted_class = outputs.argmax(-1)
        self.n_scored.add_(
            torch.tensor(
                [
                    [
                        (predicted_class == label_value)[groups == gr].long().sum()
                        for label_value in range(self.number_of_output_classes)
                    ]
                    for gr in range(self.number_of_groups)
                ]
            )
        )

    def score(self) -> float:
        return {
            gr: (self.n_scored[gr] / self.n_scored[gr].sum()).tolist()
            for gr in range(self.number_of_groups)
        }

    def __str__(self) -> str:
        return "Output distribution per group"

    @property
    def requires_data(self) -> bool:
        return True
