Using custom modules
====================

When running experiments, you can choose from the existing :doc:`registered modules </glossary>`, but
you can also define your own custom modules following the :doc:`Inspector API </api_modules>` and register them for later use in configs.

Custom runner
-------------

You might want to instantiate and run ``inspector`` in your own script, instead of relying on the cli tool provided.
You can start from this base script and extend it for your custom needs:

..  code-block:: python

    # runner.py

    import hydra
    from inspector.config import InspectorConfig

    @hydra.main(version_base="1.1", config_name="inspector", config_path=None)
    def inspect(config: InspectorConfig) -> int:
        inspector = hydra.utils.instantiate(config)
        results = inspector.run()
        inspector.save("./")

    if __name__ == "__main__":
        inspect()

This can replace the preinstalled ``inspector`` script and you can run it the same way, overriding
arguments or providing a yaml configuration file.

..  code-block:: bash

    python runner.py <cli_arguments_overrides>
    # or
    python runner.py --config-dir <config_dir> --config-name <config_name>


Custom model
------------

You can use your existing models inside the Inspector pipeline. For this, you need to implement
a wrapper of type ``InspectorModel`` that will allowing plugging your custom model into the Inspector pipeline.

1. **Model definition**. This can be any model imported from any library, or manually defined in your
existing codebase. For simplicity, let's consider the following example:

..  code-block:: python

    class FCNet(torch.nn.Module):
        def __init__(self, input_size, number_of_features, number_of_classes=2):
            self.input_size = input_size
            self.number_of_features = number_of_features
            self.number_of_classes = number_of_classes

            w, h, c = input_size
            self.fc1 = torch.nn.Linear(in_features=w * h * c, out_features=number_of_features)
            self.fc2 = torch.nn.Linear(in_features=number_of_features, out_features=number_of_features)
            self.classifier = torch.nn.Linear(
                in_features=number_of_features, out_features=number_of_classes
            )

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.classifier(x)
            return x

2. **Create dataclass config for Hydra**. Create a configuration dataclass for your custom model.
[This can optionally be registered in the config store as a schema].

..  code-block:: python

    @dataclasses.dataclass
    class FCNetConfig:
        _target_: str = "custom_model.FCNet"
        input_size: Tuple[int] = (3, 224, 224)
        number_of_features: int = 64
        number_of_classes: int = 2

3. **Implement the ``InspectorModel`` interface** as a _wrapper_ for your custom model.

..  code-block:: python

    class FCNetInspectorModel(inspector.plugins.InspectorModel):
        """Wrapper for making the FCNet model compatible with the Inspector pipeline."""

        def __init__(self, *model*: torch.nn.Module, *device*: str):
            model.to(device)
            super().__init__(model)

        def forward(self, x):
            logits = self.model(x)
            return inspector.plugins.InspectorModelOutput(logits=logits, features=None)

        def set_classification_head(self, number_of_classes: int) -> None:
            self.model.classifier = torch.nn.Linear(
                in_features=self.n_features, out_features=number_of_classes
            )
            self.model.to(self.device)
            self.model.number_of_classes = number_of_classes

        def setup(self, input_batch: torch.Tensor) -> None:
            super().setup(input_batch)

        @property
        def n_classes(self) -> int:
            return self.model.number_of_classes

        @property
        def n_features(self) -> int:
            return self.model.number_of_features

        @property
        def is_normalized(self) -> bool:
            return False

4. **Create dataclass config for the wrapper.** Separate the wrapper attributes from the actual model attributes (having two config dataclasses).

..  code-block:: python

    @dataclasses.dataclass
    class FCNetInspectorModelConfig(inspector.config.InspectorModelConfig):
        _target_: str = "custom_model.FCNetInspectorModel"
        model: FCNetConfig = dataclasses.field(default_factory=lambda: FCNetConfig())
        device: str = "cuda"

5. **Register wrapper in config store** (Optionally, add the base model so it can be used without a wrapper)

..  code-block:: python

    config_store.store(group="model", name="fcnet", node=FCNetInspectorModelConfig)

6. **Usage example:**

- Set the custom model in the config file: ``- model: fcnet``

- Instantiate Inspector using Hydra's config file: ``inspector = hydra.utils.instantiate(config)``

- In this case, ``inspector.model`` is an instance of ``FCNetInspectorModel`` and
  ``inspector.model.model`` is an instance of your custom model ``FCNet``.

Custom dataset
--------------

.. Let's see how to register a custom dataset.

1. **Dataset definition**. Implement a map-style or iterable style dataset

- Needs to output a data dict with a key ``“image”`` for the input, and one or multiple keys for
  labels or other attributes.

We show how to load ``ImageNet`` using a ``torchvision.datasets.ImageFolder`` dataset.

..  code-block:: python

    class ImageNet1k(torchvision.datasets.ImageFolder):
        def __init__(self, root: str, split: str, **kwargs) -> None:
            self.root = root
            self.split = split
            self.make_dataset_fn = lambda custom_transform: ImageNet1k(root, split, custom_transform)
            super().__init__(os.path.join(self.root, self.split), **kwargs)

        def __getitem__(self, index: int):
            image, label = super().__getitem__(index)
            return {"image": image, "label": label}

2. **Create dataclass config for Hydra**. This can be registered in the config store as a schema.

..  code-block:: python

    @dataclasses.dataclass
    class ImageNet1kConfig(inspector.config.DatasetConfig):
        _target_: str = "ImageNet1k"
        root: str = "imagenet_data/ILSVRC/Data/CLS-LOC"
        split: str = omegaconf.MISSING

3. **Register in config store**. Multiple registrations (adaptation, evaluation, corruptions).

- In order to use a dataset in the Inspector pipeline, you need to wrap it into a
  ``InspectorDataset`` object.  You need to provide a dictionary ``number_of_classes_per_attribute``
  and a ``default_attribute``.  Optionally, you can provide more meta-information about the dataset:
  ``input_size``, ``input_mean``, and ``input_std``.

For ``datasets`` and ``corruption.datasets``, register *as-dict* using ``EvaluationTransformationConfig``.

..  code-block:: python

    # Register dataset for evaluation
    transformations = inspector.config.EvaluationTransformationStackConfig()
    for group in ["datasets", "corruption.datasets"]:
        for split in ["train", "val", "test"]:
            name = f"ImageNet1k-{split}"
            node = {
                name: inspector.config.InspectorDatasetConfig(
                    dataset=ImageNet1kConfig(split=split),
                    number_of_classes_per_attribute={"label_": 1000},
                    default_attribute="label_",
                    input_size = (3, 224, 224),
                    input_mean = (0.485, 0.456, 0.406),
                    input_std = (0.229, 0.224, 0.225)
                    transformations=transformations,
                )
            }
            config_store.store(group=group, name=name, node=node)

For ``adaptation.dataset``, register *in-place* using an ``AdaptationTransformationConfig``.

..  code-block:: python

    # Register dataset for adaptation
    transformations_config = datasets_config.AdaptationTransformationStackConfig()
        node = inspector.config.InspectorDatasetConfig(
            dataset=ImageNet1kConfig(split="train"),
            number_of_classes_per_attribute={"label_": 1000},
            default_attribute="label_",
            input_size = (3, 224, 224),
            input_mean = (0.485, 0.456, 0.406),
            input_std = (0.229, 0.224, 0.225)
            transformations=transformations_config,
        )
        config_store.store(group="adaptation.dataset", name="ImageNet1k-train", node=node)

4. **Usage example**:

- Set the custom dataset in the config file:

..  code-block:: yaml

    - datasets:
        - ImageNet1k-val
        - ImageNet1k-test
    - adaptation.dataset: ImageNet1k-train

Custom evaluation metric
------------------------

1. **Metric definition**. Implement the ``Evaluation`` interface for the custom evaluation metric.
Let's consider a custom implementation of top-1 classification accuracy, which resembles the implementation
of our top-k classification accuracy in the inspector codebase.

..  code-block:: python

    class CustomTop1ClassificationAccuracy(nn.Module, Evaluation):

        n_correct: torch.Tensor
        n_scored: torch.Tensor

        def __init__(self, target_attribute: str = "default_") -> None:
            """
            Args:
                target_attribute: Attribute with respect to which we evaluate.
            """
            nn.Module.__init__(self)
            Evaluation.__init__(self, target_attribute)

        def setup(
            self,
            model: inspector.plugins.InspectorModel,
            normalization_transform: Any,
        ):
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
            predicted_class = outputs.argmax(-1)
            self.n_correct.add_((predicted_class == labels).long().sum())
            self.n_scored.add_(labels.numel())

        def score(self) -> float:
            return float(self.n_correct / self.n_scored)

        def __str__(self) -> str:
            return f"Top-1 classification accuracy"

        @property
        def requires_data(self) -> bool:
            return True

2. **Create dataclass config for Hydra**. This can be registered in the config store as a schema

..  code-block:: python

    @dataclasses.dataclass
    class CustomMetricConfig(EvaluationConfig):
        _target_: str = "custom_metric.CustomTop1ClassificationAccuracy"

3. **Register in config store**. Register as a dict so that multiple metrics can be used

..  code-block:: python

    config_store = hydra_config_store.ConfigStore.instance()
    config_store.store(
        group="evaluations", name="custom-metric", node={"custom_metric": CustomMetricConfig}
    )

4. **Usage example**:

- Set the custom metric in the config file: ``- evaluations: custom_metric`` (you can provide
  multiple metrics)

- Instantiate Inspector using Hydra's config file: ``inspector = hydra.utils.instantiate(config)``

- In this case, ``inspector.evaluations`` is an dictionary of type ``Dict[str, Evaluation]`` that
  contains the key ``"custom_metric"`` mapped to an instance of ``CustomTop1ClassificationAccuracy``
  evaluation.
