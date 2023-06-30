import dataclasses

import omegaconf


@dataclasses.dataclass
class AdversarialAttackConfig:
    """Config to create a foolbox attack instance.

    See the foolbox documentation for the list of acceptable attack names:
    https://foolbox.readthedocs.io/en/stable/modules/attacks.html .
    Optional parameters to be passed to the the attack's __init__ function can
    be added here.
    """

    _target_: str = "ood_inspector.attacks.create_foolbox_attack"
    name: str = omegaconf.MISSING


@dataclasses.dataclass
class IterativeAdversarialAttackConfig(AdversarialAttackConfig):
    """Convenience config for iterative gradient based foolbox attacks.

    Currently, these iterative attacks include:
        - L2BasicIterativeAttack
        - LinfBasicIterativeAttack
        - L2ProjectedGradientDescentAttack (a.k.a. L2PGD)
        - LinfProjectedGradientDescentAttack (a.k.a. LinfPGD or PGD)
    """

    name: str = omegaconf.MISSING
    random_start: bool = True
    steps: int = 100
    rel_stepsize: float = 0.01 / 0.3 * 40 / 100
    # TODO(cjsg): Implement the following principle:
    # rel_stepsize: float = 0.01 / 0.3 * 40 / omegaconf.II("steps")  # Doesn't work.
