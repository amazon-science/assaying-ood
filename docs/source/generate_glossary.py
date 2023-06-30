"""
This file gathers all available dataset configs from the ood_inspector, stores each dataset
config in a separate json file under CONFIGS_PATH, and then references it in the sphinx
documentation table under CSV_PATH (which is then called by glossary.rst).

Usage:
# From the Inspector root dir (the one containing the src/ and docs/ dirs)
env PYTHONPATH=src python docs/source/generate_glossary.py

Output:
Creates many .json files under CONFIGS_PATH as well as the CSV_PATH/registered_datasets.csv file
that gets called by glossary.rst to reference the different datasets and their configs in the docs.
"""
import csv
import json
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass

import hydra.core.config_store as hydra_config_store
from omegaconf import OmegaConf

import ood_inspector  # noqa F401  # Required for config_store.list to work.

MODULE_KEYS = ["datasets", "adaptation.dataset", "corruption.datasets"]

CONFIGS_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "config_options")
CSV_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "registered_datasets.csv")

DEBUG_MODE = False


@dataclass
class ConfigDetails:
    key: str
    name: str
    json: str


# def write_config_json(config_details: ConfigDetails, configs_path: str = "config_options/"):
#     with open(os.path.join(configs_path, config_details.key + ".json"), "w") as f:
#         json.dump(config_details.json, f)


def main():
    configs_details = defaultdict(list)
    config_store = hydra_config_store.ConfigStore.instance()

    # Gather available dataset options and their config files in json format
    print("Gathering all available dataset options and their configs.")
    for key in MODULE_KEYS:
        if DEBUG_MODE:
            print(config_store.list(key))

        # Nice trick to access all hydra configs. Better than extracting it from
        # ood_inspector/api/datasets/s3_webdatasets.csv
        config_files = config_store.list(key)
        for config_file in config_files:
            if config_file[-5:] != ".yaml":
                continue
            config_name = config_file[:-5]
            # Load configs and store them as dict (not DictConfig)
            config_json = OmegaConf.to_container(config_store.load(f"{key}/{config_file}").node)
            configs_details[key].append(ConfigDetails(key=key, name=config_name, json=config_json))

    if DEBUG_MODE:
        print(configs_details)

    # Save the gathered json config files under CONFIGS_PATH and link it in the documentation table
    # at CSV_PATH.
    print(
        f"Saving the gathered json files to {CONFIGS_PATH} and creating the documentation table to "
        f"reference them in the docs at {CSV_PATH}."
    )
    os.makedirs(CONFIGS_PATH, exist_ok=True)
    with open(CSV_PATH, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["Module config key", "Available options"])
        for key in MODULE_KEYS:
            line = ""
            for config_details in configs_details[key]:
                # Save config json to disk
                json_path = os.path.join(
                    CONFIGS_PATH, f"{config_details.key}_{config_details.name}.json"
                )
                relative_json_path = "/".join(json_path.split("/")[-2:])
                with open(json_path, "w") as json_file:
                    json.dump(config_details.json, json_file, indent=4)
                    # json_file.write(str(config_details.json))

                line += f":download:`{config_details.name} <{relative_json_path}>` |br| "

            # Remove last |br|
            line = line[:-6]

            # Write complete list of options for the current key
            csv_writer.writerow([key, line])


if __name__ == "__main__":
    main()
