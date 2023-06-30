from math import isnan, log
from unittest import mock

import pytest
import torch
from torch import nn

try:
    from torchvision import transforms

    from ood_inspector import attacks, evaluations
    from ood_inspector.models import inspector_base
except ModuleNotFoundError:
    pytest.skip("Not all required modules are available", allow_module_level=True)


def test_always_zero():
    metric = evaluations.AlwaysZero()
    assert metric.score() == 0


def test_number_of_parameters():
    metric = evaluations.NumberOfParameters()
    model = mock.MagicMock()
    mock_property = mock.PropertyMock(return_value=3)
    type(model).number_of_parameters = mock_property
    metric.setup(model, None)
    assert metric.score() == 3
    mock_property.assert_called_once_with()


class PyTorchLinearMLP(nn.Module):
    """A 2-layer, linear MLP for testing.
    Default usage is with images of size: (C, H, W) = (3, 2, 2).
    First layer is initialized as the identity.
    Second layer as (K-k) * sum(previous_layer), where
         K = output dimension
         k = output-index
    """

    def __init__(self, image_width: int = 2, number_of_classes: int = 2):
        super().__init__()
        number_of_neurons = 3 * image_width**2
        self._n_features = number_of_neurons
        self._n_classes = number_of_classes
        self.fc1 = nn.Linear(number_of_neurons, number_of_neurons, bias=False)
        self.fc2 = nn.Linear(number_of_neurons, number_of_classes, bias=False)
        self._initialize_layers()

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(torch.flatten(inputs, 1))

    def head(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))

    def _initialize_layers(self):
        self.fc1.weight.data = torch.eye(*self.fc1.weight.data.shape)
        self.fc2.weight.data.fill_(1.0)
        self.fc2.weight.data *= torch.arange(self._n_classes).flip(dims=(0,)).view(-1, 1)


class LinearMLP(inspector_base.InspectorModel):
    def __init__(self, number_of_classes=2):
        super().__init__(model=PyTorchLinearMLP(number_of_classes=number_of_classes))

    def setup(self) -> None:
        self.model._initialize_layers()
        self._setup = True

    def forward(self, inputs: torch.Tensor) -> inspector_base.InspectorModelOutput:
        features = self.model.features(inputs)
        logits = self.model.head(features)
        return inspector_base.InspectorModelOutput(logits=logits, features=features)

    def set_classification_head(self, number_of_classes: int):
        pass

    @property
    def n_classes(self) -> int:
        return self.model._n_classes

    @property
    def n_features(self) -> int:
        return self.model._n_features

    @property
    def is_normalized(self) -> bool:
        return False


def test_adversarial_accuracy_internal_methods():
    image = torch.arange(1 * 3 * 4 * 4).view(1, 3, 4, 4)

    # Test with mean and std being lists, 1-tuple and floats.
    for mean, std in [([1.0, 0.0, 0.0], [1.0]), ((1.0,), (1.0,)), (1.0, 1.0)]:
        normalize = transforms.Normalize(mean=mean, std=std)
        adversarial_accuracy = evaluations.AdversarialAccuracy(
            "FGSM", attack_sizes=[0.1]
        )  # FGSM = Fast Gradient Sign Method
        model = LinearMLP()
        model.eval()
        adversarial_accuracy.setup(model, normalize)

        retrieved_mean = adversarial_accuracy._mean
        retrieved_std = adversarial_accuracy._std
        assert pytest.approx(torch.tensor(mean) == retrieved_mean)
        assert pytest.approx(torch.tensor(std) == retrieved_std)

        torch_mean = torch.tensor(mean).view(-1, 1, 1)
        torch_std = torch.tensor(std).view(-1, 1, 1)
        unnormalized_image = adversarial_accuracy._unnormalize_images(image)
        normalized_image = (unnormalized_image - torch_mean) / torch_std
        assert pytest.approx(image == normalized_image)


def get_adversarial_accuracy_setup():
    # 2 batches of 2 images of size (C,H,W) = (3,2,2).
    inputs = torch.stack(
        (
            0.01 * torch.ones(3, 2, 2),  # fooled for attack_size > 0.0346.
            0.01 * torch.ones(3, 2, 2),  # fooled for attack_size > 0.0346.
            0.1 * torch.ones(3, 2, 2),  # fooled for attack_size > 0.346.
            0.1 * torch.ones(3, 2, 2),  # fooled for attack_size > 0.346.
        )
    )
    targets = torch.zeros(4, dtype=torch.long)
    normalize = transforms.Normalize(mean=0.5, std=1.0)
    return inputs, targets, normalize


def test_adversarial_accuracy():
    """
    Input image: inputs = value * torch.ones(1, 3, 2, 2)
    To compute grad: dloss/dxi is independent of i and < 0, hence:
        grad = - torch.ones(12)
        normalized_grad = (grad / grad.norm()).view(1,3,2,2) = [.2887] * 12
    FGM adversarial input is given by:
        adv_inputs = inputs + attack_size * normalized_grad
    Attack successfull if model(adv_inputs)[0] < 0 (bc outputs[1] = 0)
        model(adv_inputs) = model(inputs) + model(perturbations)
            = value * 12 - attack_size * .2887 * 12
            = (value - attack_size * .2887) * 12
    This is < 0 iff attack_size > value / .2887.
    Below, value is taken in {.01, .1} which gives pivotal attack_sizes
    {0.346, 0.0346}.
    """

    inputs, targets, normalize = get_adversarial_accuracy_setup()
    attack_sizes = [0.0, 0.34, 0.35]
    attack = attacks.create_foolbox_attack("FGM")
    metric = evaluations.AdversarialAccuracy(
        attack=attack, attack_sizes=attack_sizes, target_attribute="test_attribute"
    )
    model = LinearMLP()
    model.eval()
    metric.setup(model, normalize)
    metric.update(inputs, model(inputs), {"test_attribute": targets})
    assert pytest.approx(
        metric.score() == {"adversarial_accuracies": [1.0, 0.5, 0.0], "attack_sizes": attack_sizes}
    )


@pytest.mark.parametrize("number_of_classes", [2, 3, 4])
def test_auto_attack(number_of_classes):
    """
    Similar to adversarial accuracy test, but using auto-attack.
    """

    inputs, targets, normalize = get_adversarial_accuracy_setup()
    attack_config = dict(
        norm="L2",
        seed=42,
        verbose=False,
        version="custom",
        attacks_to_run=["apgd-ce", "apgd-dlr", "apgd-t", "fab-t"],
    )
    metric0 = evaluations.AutoAttackEvaluation(
        attack_size=0.0,
        adjust_attack_to_number_of_classes=True,
        target_attribute="test_attribute",
        **attack_config
    )
    metric1 = evaluations.AutoAttackEvaluation(
        attack_size=0.34,
        adjust_attack_to_number_of_classes=True,
        target_attribute="test_attribute",
        **attack_config
    )
    metric2 = evaluations.AutoAttackEvaluation(
        attack_size=0.35,
        adjust_attack_to_number_of_classes=True,
        target_attribute="test_attribute",
        **attack_config
    )
    model = LinearMLP(number_of_classes=number_of_classes)
    model.eval()
    for acc, metric in (1.0, metric0), (0.5, metric1), (0.0, metric2):
        metric.setup(model, normalize)
        metric.update(inputs, model(inputs).logits, {"test_attribute": targets})
        if number_of_classes == 3:
            # Replaced apgd-t (targeted dlr attack requires at least 4 classes) by apgd-dlr.
            assert metric._auto_attack.attacks_to_run == [
                "apgd-ce",
                "apgd-dlr",
                "apgd-dlr",
                "fab-t",
            ]
        elif number_of_classes == 2:
            # Replaced apgd-t and apgd-dlr by apgd-ce attacks.
            assert metric._auto_attack.attacks_to_run == ["apgd-ce", "apgd-ce", "apgd-ce", "fab-t"]
        else:
            # No attack replaced.
            assert metric._auto_attack.attacks_to_run == ["apgd-ce", "apgd-dlr", "apgd-t", "fab-t"]
        assert pytest.approx(metric.score() == acc)


def test_top1_classification_accuracy(generate_dataloader):
    inputs = torch.Tensor([0, 1, 2, 3, 4])
    outputs = torch.Tensor(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ]
    )
    targets = {"test_attribute": torch.Tensor([0, 0, 0, 1, 1])}

    metric = evaluations.ClassificationAccuracy(target_attribute="test_attribute", top_k=1)
    metric.setup(None, None)
    metric.update(inputs, outputs, targets)
    assert pytest.approx(metric.score()) == 3 / 5


def test_top3_classification_accuracy(generate_dataloader):
    inputs = torch.Tensor([0, 1, 2, 3, 4])
    outputs = torch.tile(torch.Tensor([1, 1, 1, 0, 0, 0]).unsqueeze(0), (5, 1))
    targets = {"test_attribute": torch.Tensor([0, 5, 1, 3, 2])}
    metric = evaluations.ClassificationAccuracy(target_attribute="test_attribute", top_k=3)
    metric.setup(None, None)
    # The top-3 predicted classes of this model are always [0, 1, 2].
    metric.update(inputs, outputs, targets)
    assert pytest.approx(metric.score()) == 3 / 5


def test_top3_classification_accuracy_invalid_inputs():
    # outputs.shape[-1] = 2, hence top_3 parameters exceeds number of classes.
    # This should return the score = nan.
    inputs = torch.Tensor([0, 1, 2, 3, 4])
    outputs = torch.tile(torch.Tensor([1, 0]).unsqueeze(0), (5, 1))
    targets = {"test_attribute": torch.Tensor([0, 5, 1, 3, 2])}

    metric = evaluations.ClassificationAccuracy(top_k=3, target_attribute="test_attribute")
    metric.setup(None, None)
    metric.update(inputs, outputs, targets)
    assert isnan(metric.score())


def test_ece_valid_inputs(generate_dataloader):
    inputs = torch.Tensor([0, 1])
    targets = {"test_attribute": torch.Tensor([0, 0])}

    logits = torch.Tensor([[2, 2], [0, 100]])
    probabilities = torch.Tensor([[0.5, 0.5], [0, 1]])
    # Correct predictions: 0, 0.
    # Accuracy bins: NaN, 0, 0.
    # Confidence bins: NaN, 0.5, 1.
    # ECE: 0.5 * 1/2 + 1 * 1/2 = 0.75.
    model_logits = mock.Mock(
        is_normalized=False,
        return_value=inspector_base.InspectorModelOutput(logits=logits, features=None),
    )
    metric = evaluations.ExpectedCalibrationError(number_bins=3, target_attribute="test_attribute")
    metric.setup(model_logits, None)
    metric.update(inputs, model_logits(inputs).logits, targets)
    assert pytest.approx(metric.score()) == 0.75

    model_probabilities = mock.Mock(
        is_normalized=True,
        return_value=inspector_base.InspectorModelOutput(logits=probabilities, features=None),
    )
    metric = evaluations.ExpectedCalibrationError(number_bins=3, target_attribute="test_attribute")
    metric.setup(model_probabilities, None)
    metric.update(inputs, model_probabilities(inputs).logits, targets)
    assert pytest.approx(metric.score()) == 0.75


@pytest.mark.parametrize(
    "metric", [evaluations.ExpectedCalibrationError, evaluations.NegativeLogLikelihood]
)
def test_unnormalized_inputs(metric):
    inputs = torch.Tensor([0, 1])
    targets = {"test_attribute": torch.Tensor([0, 0])}

    logits = torch.Tensor([[2, 2], [0, 2]])
    model_logits = mock.Mock(
        is_normalized=True,
        return_value=inspector_base.InspectorModelOutput(logits=logits, features=None),
    )
    metric_instance = metric(target_attribute="test_attribute")
    metric_instance.setup(model_logits, None)

    # If unnormalized logits are passed, an error should be thrown.
    with pytest.raises(ValueError):
        metric_instance.update(inputs, model_logits(inputs).logits, targets)


def test_ece_binary_classification(generate_dataloader):
    inputs = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
    targets = {"test_attribute": torch.Tensor([1, 1, 1, 0, 1, 0, 0, 0])}
    number_inputs = len(targets["test_attribute"])

    number_bins = 10
    probabilities_class1 = torch.Tensor([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])[:, None]
    prediction_probabilities = torch.cat((probabilities_class1, 1 - probabilities_class1), dim=-1)
    # Confidences: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85].
    # Predicted classes: [1, 0, 0, 1, 1, 0, 1, 1].

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ..., [0.9, 1).
    correct_bin_counts = [0, 0, 0, 0, 0, 2, 3, 1, 2, 0]
    correct_bin_sum = [0, 0, 0, 0, 0, 1, 2, 0, 2, 0]
    bin_probability_sums = [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71, 0.81 + 0.85, 0]

    correct_ece = 0.0
    accuracy_bins = [0.0] * number_bins
    confidence_bins = [0.0] * number_bins
    for i in range(number_bins):
        if correct_bin_counts[i] > 0:
            accuracy_bins[i] = correct_bin_sum[i] / correct_bin_counts[i]
            confidence_bins[i] = bin_probability_sums[i] / correct_bin_counts[i]
            correct_ece += (
                correct_bin_counts[i] / number_inputs * abs(accuracy_bins[i] - confidence_bins[i])
            )

    metric = evaluations.ExpectedCalibrationError(
        number_bins=number_bins, target_attribute="test_attribute"
    )
    # This model predicts with the given prediction_probabilities.
    model = mock.Mock(
        is_normalized=True,
        return_value=inspector_base.InspectorModelOutput(
            logits=prediction_probabilities, features=None
        ),
    )
    metric.setup(model, None)
    metric.update(inputs, model(inputs).logits, targets)

    assert pytest.approx(metric.score()) == correct_ece
    correct_bin_counts = torch.as_tensor(correct_bin_counts, dtype=metric.bin_counts.dtype)
    assert all(torch.isclose(metric.bin_counts, correct_bin_counts))
    correct_bin_sum = torch.as_tensor(correct_bin_sum, dtype=metric.accuracy_bins.dtype)
    assert all(torch.isclose(metric.accuracy_bins, correct_bin_sum))
    bin_probability_sums = torch.as_tensor(bin_probability_sums, dtype=metric.confidence_bins.dtype)
    assert all(torch.isclose(metric.confidence_bins, bin_probability_sums))


@pytest.mark.parametrize(
    "predictions,groups,expected",
    [
        (
            torch.Tensor([0, 1, 0, 1]).long(),
            torch.Tensor([True, True, False, False]).bool(),
            0.0,
        ),
        (
            torch.Tensor([0, 1, 0, 1, 1]).long(),
            torch.Tensor([True, True, False, False, False]).bool(),
            2.0 / 25,
        ),
        (
            torch.Tensor([0, 0, 0, 0, 1, 1]).long(),
            torch.Tensor([True, True, False, False, False, False]).bool(),
            2.0 / 9,
        ),
        (
            torch.Tensor([0, 0, 0, 0, 1, 1]).long(),
            torch.Tensor([False, False, False, False, False, False]).bool(),
            0.0,
        ),
    ],
)
def test_demographic_parity(predictions, groups, expected):
    # In the first testcase, the predictions are balanced.
    # In the second testcase, the distribution of predictions is [2, 3], so the frequency is
    # [2/5, 3/5]. In one environment we have distribution [0.5, 0.5], in the other [1/3, 2/3].
    # So the total variation for the first env. is 0.5 * sum|[0.4, 0.6] - [0.5, 0.5]| = 1/10,
    # for the second is 0.5 * sum|[2/5, 3/5] - [1/3, 2/3]| = 1/15. The distribution of groups is
    # 2/5 and 3/5, so the average is 1/10*2/5 + 1/15*3/5 = 2/25.
    # In the third testcase, the distribution of predictions is [4, 2], so the frequency is
    # [2/3, 1/3]. In one environment we have distribution [1, 0], in the other [1/2, 1/2].
    # So the total variation for the first env. is 0.5 * sum|[2/3, 1/3] - [1, 0]| = 1/3,
    # for the second is 0.5 * sum|[2/3, 1/3] - [1/2, 1/2]| = 1/6. The distribution of groups is
    # 1/3 and 2/3, so the average is 1/3*1/3 + 1/6*2/3 = 2/9.
    # In the fourth testcase, there is only one environment.

    metric = evaluations.DemographicParityInferredGroups(number_of_samples=len(predictions))

    unfairness = metric.demographic_parity_with_groups(predictions, groups, 2)

    assert pytest.approx(unfairness) == expected


def test_negative_log_likelihood(generate_dataloader):
    inputs = torch.Tensor([0, 1])
    probability_output = torch.Tensor([0.1, 0.3, 0.6])
    targets = {"test_attribute": torch.Tensor([1, 2]).long()}
    expected = -(log(0.3) + log(0.6)) / 2

    metric = evaluations.NegativeLogLikelihood(target_attribute="test_attribute")
    # This model predicts class probabilities `probability_output` all the time.
    model = mock.Mock(
        is_normalized=True,
        return_value=inspector_base.InspectorModelOutput(logits=probability_output, features=None),
    )
    metric.setup(model, None)
    metric.update(inputs, torch.tile(probability_output.unsqueeze(0), (2, 1)), targets)

    assert pytest.approx(metric.score()) == expected


def test_top1_classification_accuracy_per_group(generate_dataloader):
    inputs = torch.Tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    outputs = torch.Tensor(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ]
    )
    targets = {
        "test_attribute": torch.Tensor([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
        "group_attribute": torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }

    metric = evaluations.ClassificationAccuracyPerGroup(
        target_attribute="test_attribute", group_attribute="group_attribute", top_k=1
    )
    metric.setup(None, None, {"test_attribute": 2, "group_attribute": 2})
    metric.update(inputs, outputs, targets)
    assert pytest.approx(metric.score()[0]) == 3 / 5 and pytest.approx(metric.score()[1]) == 3 / 5


def test_top3_classification_accuracy_per_group(generate_dataloader):
    inputs = torch.Tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    outputs = torch.tile(torch.Tensor([1, 1, 1, 0, 0, 0]).unsqueeze(0), (10, 1))
    targets = {
        "test_attribute": torch.Tensor([0, 5, 1, 3, 2, 0, 5, 1, 3, 2]),
        "group_attribute": torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }
    metric = evaluations.ClassificationAccuracyPerGroup(
        target_attribute="test_attribute", group_attribute="group_attribute", top_k=3
    )
    metric.setup(None, None, {"test_attribute": 2, "group_attribute": 2})
    # The top-3 predicted classes of this model are always [0, 1, 2].
    metric.update(inputs, outputs, targets)
    assert pytest.approx(metric.score()[0]) == 3 / 5 and pytest.approx(metric.score()[1]) == 3 / 5


def test_output_distribution_per_group(generate_dataloader):
    inputs = torch.Tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    outputs = torch.Tensor(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    targets = {
        "test_attribute": torch.Tensor([0, 0, 2, 2, 0, 0, 0, 0, 1, 1]),
        "group_attribute": torch.Tensor([0, 0, 2, 2, 1, 1, 1, 1, 1, 1]),
    }

    metric = evaluations.OutputDistributionPerGroup(
        target_attribute="test_attribute", group_attribute="group_attribute"
    )
    metric.setup(None, None, {"test_attribute": 3, "group_attribute": 3})
    metric.update(inputs, outputs, targets)
    assert (
        pytest.approx(metric.score()[0]) == [1, 0, 0]
        and pytest.approx(metric.score()[1]) == [1 / 3, 1 / 3, 1 / 3]
        and pytest.approx(metric.score()[2]) == [0, 1 / 2, 1 / 2]
    )
