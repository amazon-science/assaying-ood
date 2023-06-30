import logging
from typing import Any, Dict, List, Optional

import autoattack
import foolbox.attacks

from ood_inspector.models import inspector_base

logger = logging.getLogger(__name__)


def create_foolbox_attack(name: str, **kwargs: Any) -> foolbox.attacks.base.Attack:
    """Creates a foolbox attack instance with optional parameters kwargs.

    Args:
      name: The name of a foolbox attack. You can find the list of all foolbox
        attacks in the foolbox API here:
        https://foolbox.readthedocs.io/en/stable/modules/attacks.html .
      kwargs: Additional arguments to be passed when instantiating foolbox attack.
    """
    try:
        return getattr(foolbox.attacks, name)(**kwargs)
    except AttributeError:
        raise AttributeError(
            f"Unknown foolbox attack {name}. See foolbox docs for the list of "
            "acceptable attacks: "
            "https://foolbox.readthedocs.io/en/stable/modules/attacks.html ."
        )
    except TypeError as e:
        raise TypeError(
            f"{name}'s {e}. See foolbox docs for the list of acceptable "
            f"arguments: "
            "https://foolbox.readthedocs.io/en/stable/modules/attacks.html ."
        )


def create_auto_attack(
    model: inspector_base.ModelWrapperForLogits,
    attack_customization_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    **attack_initialization_kwargs: Any,
) -> autoattack.AutoAttack:
    """Creates an AutoAttack instance from https://arxiv.org/abs/2003.01690

    Arguments
    ---------
        attack_initialization_kwargs:
            Should contain all arguments to be passed to the AutoAttack constructor. Currently,
            these arguments (and defaults) are: norm ('Linf'), eps (.3), seed (None), verbose
            (True), attacks_to_run ([]), version ('standard'), is_tf_model (False), device ('cuda'),
            log_path (None).

        attack_customization_parameters:
            An optional dict containing all parameters to pass to customize the auto-attack instance
            after initialization. It maps attack names (as in ``attacks_to_run``) to their
            attack-specific parameters (which are set separately, after the instanciation of
            AutoAttack). It's typical use would be with custom auto-attacks (i.e., having
            ``version='custom'`` in the ``attack_initialization_kwargs``. Example of a valid
            attack_initialization_kwargs:
            ::

                >>> {
                ...    apgd: {n_restarts: 1},
                ...    fab: {n_restarts: 1, n_target_classes: 6}
                ... }


    References
    ----------
        * Original paper of the attack: https://arxiv.org/abs/2003.01690
        * Also used in RobustBench:
            * paper: https://arxiv.org/abs/2010.09670
            * website: https://robustbench.github.io/
    """
    auto_attack = autoattack.AutoAttack(model, **attack_initialization_kwargs)
    attack_customization_parameters = attack_customization_parameters or dict()
    for simple_attack_name in attack_customization_parameters:
        simple_attack = getattr(auto_attack, simple_attack_name)
        simple_attack_parameter_dict = attack_customization_parameters[simple_attack_name]
        for parameter_name, parameter_value in simple_attack_parameter_dict.items():
            setattr(simple_attack, parameter_name, parameter_value)
    return auto_attack


def _replace_attack(attacks_to_run: List[str], old_attack_name: str, new_attack_name: str) -> None:
    for i in range(len(attacks_to_run)):
        if attacks_to_run[i] == old_attack_name:
            attacks_to_run[i] = new_attack_name


def adjust_autoattack_to_number_of_classes(
    n_classes: int, auto_attack: autoattack.AutoAttack
) -> None:
    """Checks that there are enough output classes for all auto-attack's attacks to run, and
    replaces them if needed.

    Some attacks used by auto-attack require a minimal amount of classes (at least 4 for
    apgd-dlr-targeted, at least 3 for apgd-dlr). This function replaces these attacks by others if
    needed. For targeted attacks, it also ensures that the n_target_classes attribute is at most
    equal to the number of incorrect classes (i.e., total number of classes - 1).
    """
    # TODO(cjsg): Ideally, this function should would be placed inside of create_auto_attack and get
    # called if an appropriate flag (e.g., adjust_attack_to_number_of_parameters) is set to True,
    # using the model's n_classes attribute. However, since we sometimes still initialize this
    # attribute to None, it is sager for now to call this function inside the AutoAttackEvaluation.
    assert isinstance(n_classes, int) and n_classes > 1
    # No need to check if auto_attack has attributes apgd_targeted or farb, because all auto-attacks
    # get initialized with it, even if they don't use it.
    auto_attack.apgd_targeted.n_target_classes = min(
        n_classes - 1, auto_attack.apgd_targeted.n_target_classes
    )
    auto_attack.fab.n_target_classes = min(n_classes - 1, auto_attack.fab.n_target_classes)
    if n_classes == 3 and "apgd-t" in auto_attack.attacks_to_run:
        if auto_attack.apgd_targeted.loss == "dlr-targeted":
            logger.warning(
                "The apgd attack with dlr-targeted loss requires at least 4 output classes, but "
                "current network has only 3. Replacing apgd-t by apgd-dlr in attacks_to_run."
            )
            _replace_attack(auto_attack.attacks_to_run, "apgd-t", "apgd-dlr")
    elif n_classes == 2:
        if "apgd-t" in auto_attack.attacks_to_run:
            logger.warning(
                "Network has 2 output classes only. Removing apgd-t from attacks_to_run "
                "and appending apgd-ce attack to it instead."
            )
            _replace_attack(auto_attack.attacks_to_run, "apgd-t", "apgd-ce")
        if "apgd-dlr" in auto_attack.attacks_to_run:
            logger.warning(
                "The apgd-dlr attack requires at least 3 output classes, but network has only 2. "
                "Removing apgd-dlr from attacks_to_run and appending apgd-ce attack instead."
            )
            _replace_attack(auto_attack.attacks_to_run, "apgd-dlr", "apgd-ce")
