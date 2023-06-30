import dataclasses
import json
import logging
import os
from typing import Any, Dict, Optional

import dill
import torch
import tqdm

import ood_inspector.adaptation
import ood_inspector.corruption
from ood_inspector import inspector, torch_core
from ood_inspector.evaluations import FairnessEvaluation
from ood_inspector.models import inspector_base

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Inspector:
    """Inspector class."""

    model: inspector_base.InspectorModel
    # TODO(hornmax): Should add type specification here, this requires a metric base class.  Saving
    # for a separate commit.
    evaluations: Dict
    adaptation: ood_inspector.adaptation.ModelAdaptation = ood_inspector.adaptation.NoAdaptation()
    corruption: ood_inspector.corruption.Corruption = ood_inspector.corruption.NoCorruption()
    datasets: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # TODO(cjsg): enable passing batch_size key to dataset (instead of
    # dataloader) when webdataset is used.
    dataloader: Any = torch_core.TorchDataLoader()
    results: Dict[str, Any] = dataclasses.field(default_factory=lambda: {})
    s3_output_path: Optional[str] = None
    save_inspector: bool = False

    def __post_init__(self):
        # Check if the conditions of evaluations are satisfied.
        for name, evaluation in self.evaluations.items():
            if evaluation.requires_data and not self.datasets:
                raise ValueError(
                    f"Evaluation {name} ({evaluation}) requires a dataset but none given."
                )

    def run(self) -> Dict:
        self.add_corrupted_datasets()
        self.adapt()
        self.eval()
        return self.results

    @torch.no_grad()
    def eval(self) -> None:
        # Split dictionary into evaluations which require data access and those which do not.
        evaluations_with_data = list(
            filter(lambda elem: elem[1].requires_data, self.evaluations.items())
        )
        evaluations_no_data = list(
            filter(lambda elem: not elem[1].requires_data, self.evaluations.items())
        )
        for name, evaluation in evaluations_no_data:
            evaluation.setup(self.model, None)
            evaluation_result = evaluation.score()
            self.results[name] = evaluation_result
            logging.info(f"{name} : {evaluation_result}")

        self.model.eval()
        device = self.model.device
        for dataset_name, dataset in self.datasets.items():
            logging.info(f"Start evaluation on {dataset_name}")
            # An evaluation can only be applied to datasets whose attributes contain the
            # evaluation's target attribute and, for a fairness evaluation, the evaluation's group
            # attribute. Hence, we start by collecting all evaluations that are applicable to
            # the current dataset and store them in `evaluations_on_dataset`.
            evaluations_on_dataset = {}
            for name, evaluation in evaluations_with_data:
                if evaluation.target_attribute == "default_":
                    evaluation.target_attribute = dataset.default_attribute
                if evaluation.target_attribute not in dataset.attributes:
                    log.warning(
                        f"Target attribute {evaluation.target_attribute} required "
                        f"for evaluating {name} is not in dataset"
                    )
                    continue
                if isinstance(evaluation, FairnessEvaluation):
                    if evaluation.group_attribute == "default_":
                        evaluation.group_attribute = dataset.default_attribute
                    if evaluation.group_attribute not in dataset.attributes:
                        log.warning(
                            f"Group attribute {evaluation.group_attribute} required "
                            f"for evaluating {name} is not in dataset"
                        )
                        continue
                evaluations_on_dataset[name] = evaluation
                if isinstance(evaluation, FairnessEvaluation):
                    evaluation.setup(
                        self.model, dataset.normalization, dataset.number_of_classes_per_attribute
                    )
                else:
                    evaluation.setup(self.model, dataset.normalization)

            dataloader = self.dataloader(dataset)
            iterator_with_progress = tqdm.tqdm(dataloader, total=len(dataloader), desc="Predicting")
            for sample in iterator_with_progress:
                inputs = sample["image"]
                all_labels = {
                    attribute: sample[attribute].to("cpu") for attribute in dataset.attributes
                }
                outputs = self.model(inputs.to(device))
                outputs = outputs.logits.to("cpu")
                inputs = inputs.to("cpu")
                # Update evaluators with each batch.

                for name, evaluation in evaluations_on_dataset.items():
                    # TODO(flwenzel): Think about a cleaner way of skipping corruptions datasets.
                    # Preferably this would be handled plugin-based to not convolute our codebase
                    # with such special case options.
                    if "Corrupted" in dataset_name and not evaluation.apply_on_corrupted_datasets:
                        continue
                    else:
                        evaluation.update(inputs, outputs, all_labels)

            # Compute scores after full pass over the data.
            for name, evaluation in evaluations_on_dataset.items():
                evaluation_result = evaluation.score()
                if name not in self.results.keys():
                    self.results[name] = {}
                self.results[name][dataset_name] = evaluation_result
                logging.info(f"{name} - {dataset_name}: {evaluation_result}")

    def adapt(self) -> None:
        self.model = self.adaptation.fit(self.model)

    def add_corrupted_datasets(self) -> None:
        if not self.corruption.datasets and self.datasets:
            # Apply corruptions to all evaluation datasets.
            self.corruption.datasets = self.datasets
        if self.corruption.datasets:
            # Append corrupted datasets to list of evaluation datasets.
            corrupted_datasets = self.corruption.apply_corruptions_to_datasets()
            self.datasets.update(corrupted_datasets)

    def save(self, savedir: str, **additional_data) -> None:
        """Saves the results and inspector."""
        if self.save_inspector:
            inspector_path = os.path.join(savedir, "inspector.pkl")
            logging.info(f"Saving inspector to {inspector_path}")
            with open(inspector_path, "wb") as inspector_file:
                dill.dump(self, inspector_file)

        result_path = os.path.join(savedir, "results.json")
        logging.info(f"Saving results to {result_path}")
        result_dict = {"results": self.results, **additional_data}

        with open(result_path, "w") as result_file:
            json.dump(result_dict, result_file, indent=2)


def load(loaddir: str = os.getcwd()) -> inspector.Inspector:
    inspector_path = os.path.join(loaddir, "inspector.pkl")
    with open(inspector_path, "rb") as inspector_file:
        return dill.load(inspector_file)
