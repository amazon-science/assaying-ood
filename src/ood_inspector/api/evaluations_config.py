import dataclasses
import math
from typing import Any, Dict, List, Optional

import hydra.core.config_store as hydra_config_store
import omegaconf

from ood_inspector.api import attacks_config


@dataclasses.dataclass
class EvaluationConfig:
    target_attribute: str = "default_"


@dataclasses.dataclass
class FairnessEvaluationConfig(EvaluationConfig):
    group_attribute: str = "default_"


@dataclasses.dataclass
class MockEvaluationConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.AlwaysZero"


@dataclasses.dataclass
class NumberOfParametersConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.NumberOfParameters"


@dataclasses.dataclass
class ClassificationAccuracyConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.ClassificationAccuracy"
    top_k: int = 1


@dataclasses.dataclass
class ClassificationAccuracyPerGroupConfig(FairnessEvaluationConfig):
    _target_: str = "ood_inspector.evaluations.ClassificationAccuracyPerGroup"
    top_k: int = 1


@dataclasses.dataclass
class AdversarialAccuracyConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.AdversarialAccuracy"
    attack: attacks_config.AdversarialAttackConfig = attacks_config.AdversarialAttackConfig()
    attack_sizes: List[float] = omegaconf.MISSING
    number_of_samples: Optional[int] = None
    skip_corruption_datasets: bool = False


@dataclasses.dataclass
class AutoAttackEvaluationConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.AutoAttackEvaluation"
    attack_size: float = omegaconf.MISSING
    number_of_samples: Optional[int] = None
    skip_corruption_datasets: bool = False
    adjust_attack_to_number_of_classes: bool = True
    norm: str = omegaconf.MISSING
    verbose: bool = False


@dataclasses.dataclass
class StandardLinfAutoAttackEvaluationConfig(AutoAttackEvaluationConfig):
    norm: str = "Linf"
    version: str = "standard"


@dataclasses.dataclass
class StandardL2AutoAttackEvaluationConfig(AutoAttackEvaluationConfig):
    norm: str = "L2"
    version: str = "standard"


@dataclasses.dataclass
class CustomAutoAttackEvaluationConfig(AutoAttackEvaluationConfig):
    version: str = "custom"
    attacks_to_run: List[str] = omegaconf.MISSING
    attack_customization_parameters: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory=lambda: {"apgd": {"n_restarts": 1, "n_iter": 100}}
    )


@dataclasses.dataclass
class CustomLinfAutoAttackEvaluationConfig(CustomAutoAttackEvaluationConfig):
    attacks_to_run: List[str] = ("apgd-ce", "apgd-dlr")
    norm: str = "Linf"


@dataclasses.dataclass
class CustomL2AutoAttackEvaluationConfig(CustomAutoAttackEvaluationConfig):
    attacks_to_run: List[str] = ("apgd-ce", "apgd-dlr")
    norm: str = "L2"


@dataclasses.dataclass
class LinfAPGDCEAutoAttackEvaluationConfig(CustomAutoAttackEvaluationConfig):
    attacks_to_run: List[str] = ("apgd-ce",)
    norm: str = "Linf"


@dataclasses.dataclass
class LinfAPGDDLRAutoAttackEvaluationConfig(CustomAutoAttackEvaluationConfig):
    attacks_to_run: List[str] = ("apgd-dlr",)
    norm: str = "Linf"


@dataclasses.dataclass
class ExpectedCalibrationErrorConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.ExpectedCalibrationError"
    number_bins: int = 10


@dataclasses.dataclass
class DemographicParityInferredGroupsConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.DemographicParityInferredGroups"
    number_of_samples: Optional[int] = None


@dataclasses.dataclass
class NegativeLogLikelihoodConfig(EvaluationConfig):
    _target_: str = "ood_inspector.evaluations.NegativeLogLikelihood"


@dataclasses.dataclass
class OutputDistributionPerGroupConfig(FairnessEvaluationConfig):
    _target_: str = "ood_inspector.evaluations.OutputDistributionPerGroup"


# Naming convention: snake case of config class without `Config` suffix.
_EVALUATIONS = {
    "mock_evaluation": MockEvaluationConfig,
    "number_of_parameters": NumberOfParametersConfig,
    "classification_accuracy": ClassificationAccuracyConfig,
    "classification_accuracy_per_group": ClassificationAccuracyPerGroupConfig,
    "expected_calibration_error": ExpectedCalibrationErrorConfig,
    "adversarial_accuracy": AdversarialAccuracyConfig,
    "auto_attack": AutoAttackEvaluationConfig,
    "demographic_parity_inferred_groups": DemographicParityInferredGroupsConfig,
    "negative_log_likelihood": NegativeLogLikelihoodConfig,
    "output_distribution_per_group": OutputDistributionPerGroupConfig,
}

config_store = hydra_config_store.ConfigStore.instance()
for name, config_object in _EVALUATIONS.items():
    # Store the schema (for use in yaml files).
    config_store.store(group="schemas/evaluations", name=name, node=config_object)
    # Store the config (for command line use).
    config_store.store(group="evaluations", name=name, node={name: config_object})


# Convenience groups for ease of use.
config_store.store(
    group="evaluations",
    name="top_k_classificaton_accuracy",
    node={
        "top1_classification_accuracy": ClassificationAccuracyConfig(top_k=1),
        "top5_classification_accuracy": ClassificationAccuracyConfig(top_k=5),
        "top10_classification_accuracy": ClassificationAccuracyConfig(top_k=10),
    },
)

for name, config_object, attack_size_factor in [
    ("linf_auto_attack", StandardLinfAutoAttackEvaluationConfig, 1.0),
    ("l2_auto_attack", StandardL2AutoAttackEvaluationConfig, 224 * math.sqrt(3.0)),
    ("linf_custom_auto_attack", CustomLinfAutoAttackEvaluationConfig, 1.0),
    ("l2_custom_auto_attack", CustomL2AutoAttackEvaluationConfig, 224 * math.sqrt(3.0)),
    ("linf_apgd_ce_auto_attack", LinfAPGDCEAutoAttackEvaluationConfig, 1.0),
    ("linf_apgd_dlr_auto_attack", LinfAPGDDLRAutoAttackEvaluationConfig, 1.0),
]:
    config_store.store(group="schemas/evaluations", name=name, node=config_object)
    config_store.store(group="evaluations", name=name, node={name: config_object})

    number_of_samples = 1000
    skip_corruption_datasets = True
    attacks_config_dict = dict()
    for attack_size in [0.005, 0.05, 0.1, 0.15, 0.3, 0.5]:
        scaled_attack_size = (attack_size / 255.0) * attack_size_factor
        attacks_config_dict[f"size{attack_size}_{name}_{number_of_samples}"] = config_object(
            attack_size=scaled_attack_size,
            number_of_samples=number_of_samples,
            skip_corruption_datasets=skip_corruption_datasets,
        )
    config_store.store(group="evaluations", name=f"{name}s", node=attacks_config_dict)

linf_attack_sizes = [size / 255.0 for size in [5e-3, 5e-2, 1e-1, 1.5e-1, 3e-1, 5e-1]]
config_store.store(
    group="evaluations",
    name="foolbox_attacks",
    node={
        "adversarial_accuracy_pgd_100": AdversarialAccuracyConfig(
            attack=attacks_config.IterativeAdversarialAttackConfig(
                name="PGD", random_start=True, steps=100, rel_stepsize=(0.01 / 0.3 * 40 / 100)
            ),
            attack_sizes=linf_attack_sizes,
        ),
        "adversarial_accuracy_l2pgd_100": AdversarialAccuracyConfig(
            attack=attacks_config.IterativeAdversarialAttackConfig(
                name="L2PGD", random_start=True, steps=100, rel_stepsize=(0.01 / 0.3 * 40 / 100)
            ),
            attack_sizes=[size * 224 * math.sqrt(3) for size in linf_attack_sizes],
        ),
    },
)
