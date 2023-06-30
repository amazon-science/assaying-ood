#!/usr/bin/env python3

# Run the inspector
"""Run inspector

An example call could look like:
    python bin/run.py output_path=output/goes/here \
        +datasets=mock \
        +evaluations=mock \
        +models=mock
"""

import importlib
import logging
import os
import sys
from typing import Optional

import hydra
import omegaconf
import yaml
from hydra.core import hydra_config
from hydra.core.override_parser.overrides_parser import OverridesParser

import ood_inspector.utils as inspector_utils
from ood_inspector.api import inspector_config

LOGGER = logging.getLogger(__name__)


def load_custom_plugins(user_directory: Optional[str] = None) -> None:
    if not user_directory:
        return

    user_directory = os.path.abspath(user_directory)
    package_path, package_name = os.path.split(user_directory)

    sys.path.insert(0, package_path)
    importlib.import_module(package_name)

    for filename in os.listdir(user_directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            importlib.import_module(f"{package_name}.{module_name}")
            print(f"Found user module: {module_name}")


def savedir() -> str:
    """Returns hydra run or sweep dir."""
    hydra_cfg = hydra_config.HydraConfig.get()
    if omegaconf.OmegaConf.is_missing(hydra_cfg["job"], "num"):
        return hydra_cfg.run.dir
    # Return the multi-run directory.
    return os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)


def get_key_from_override(override):
    key = override.get_key_element()
    if override.is_delete() or override.is_add():
        return key[1:]
    elif override.is_force_add():
        return key[2:]
    else:
        return key


def overrides() -> dict:
    """Get overrides as dictionary.

    This allows us to parse the results in a more easy manner as it encodes the settings we apply to
    the hydra config tree in a easy to parse manner.
    """
    hydra_cfg = hydra_config.HydraConfig.get()
    task_overrides = hydra_cfg.overrides.task
    override_parser = OverridesParser.create()
    overrides = override_parser.parse_overrides(task_overrides)
    return {get_key_from_override(override): override.value() for override in overrides}


@hydra.main(version_base="1.1", config_name="inspector", config_path=None)
def inspect(config: inspector_config.InspectorConfig) -> int:
    return_code = 0
    try:
        LOGGER.info("Starting Inspector")
        LOGGER.debug(omegaconf.OmegaConf.to_yaml(config))
        inspector = hydra.utils.instantiate(config)
        LOGGER.debug(inspector)
        inspector.run()
        # Hydra takes care of working dir management. Thus we only need to save results into the
        # current working directory.
        inspector.save(
            ".",
            config=yaml.load(omegaconf.OmegaConf.to_yaml(config), Loader=yaml.SafeLoader),
            overrides=overrides(),
        )
    except Exception as e:
        LOGGER.exception(e)
        return_code = 1
    if config.s3_output_path:
        output_dir: str = savedir()
        s3_path = f"{config.s3_output_path}/{output_dir}"
        logging.info(f"Saving run outputs to s3 path {s3_path}.")
        inspector_utils.flush_logs()
        inspector_utils.copy_run_output_to_s3(s3_path)

    # Return with an error code in case we caught an error during execution.
    return return_code


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    load_custom_plugins(os.environ.get("INSPECTOR_CUSTOM_MODULES_PATH"))
    inspect()
