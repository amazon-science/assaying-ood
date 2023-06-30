# Inspector

**A highly configurable framework for assaying the performance of machine learning pipelines.**

``Inspector`` is a library for evaluating robustness, generalization and fairness properties of neural networks with the goal of speeding up and standardizing the testing process of new results.

## Features
- |:electric_plug:| Plugin-based architecture allowing to replace any component in the pipeline.
- |:package:| Wide range of modules ready to be used out-of-the-box:
    - **Models**: All ``timm`` and ``torchvision`` models are supported.
    - **Datasets**: DomainBed collection, WILDS collection, ImageNet.
    - **Augmentations**:``augly_transforms`` and ``timm.data.auto_augment`` are supported.
    - **Corruptions**: ``imagenet_c`` corruptions supported.
    - **Adaptation strategies**: Finetuning the whole model or only the classification head.
    - **Optimizers**: All ``torch.optim`` optimizers are supported.
    - **Learning rate schedulers**: ``torch.optim.lr_scheduler`` supported.
    - **Metrics**: classification accuracy, adversarial accuracy, auto-attack evaluation, reliability diagram, expected calibration error, demographic parity inferred groups, negative log-likelihood, classification accuracy per group, output distribution per group, number of parameters.
- |:wrench:| You can easily plug your custom modules into the pipeline.
- |:dragon:| Integrated with ``hydra`` for flexible configuration of all modules.


## Installation

```bash
    # Install library and cli `inspector` script
    pip install git+https://github.com/amazon-science/assaying-ood.git

    # Download all datasets locally
    inspector download-datasets  # TODO(armannic): move download script here
```

## Usage examples

Run inspector from the CLI script installed along the library.
```bash
inspector +model=timm_pretrained_resnet18 \
    adaptation=finetune \
    +adaptation.number_of_epochs=1 \
    +adaptation.optimizer.defaults.lr=0.001 \
    +adaptation.dataset=S3DomainBed-PACS-cartoon \
    +datasets=S3DomainBed-PACS-sketch \
    +evaluations=classification_accuracy
```

The same experiment can be produced by passing a yaml config file:

```bash
inspector --config-dir . --config-name example.yaml
```

``` yaml
# example.yaml

defaults:
  - inspector
  - datasets: S3DomainBed-PACS-cartoon
  - adaptation.dataset: S3DomainBed-PACS-sketch
  - model: timm_pretrained_resnet18
  - evaluations: classification_accuracy
  - override adaptation: finetune

adaptation:
  optimizer:
    defaults:
      lr: 0.01
  number_of_epochs: 1
```

<!-- See the complete list of registered modules available in the [Registered modules](/docs/build/html/registered_datasets.html#registered-datasets) section.

For more details about composing and overriding configuration values checkout Hydra's [documentation](https://hydra.cc/docs/intro/).

You can customize the inspector runner starting from the script [here](link_runner).

More configuration examples are available [here](link_configs). -->

<!-- You can implement your own runner and instantiate Inspector:

```python
@hydra.main(version_base="1.1", config_name="inspector", config_path=None)
def inspect(config: inspector_config.InspectorConfig) -> int:
    inspector = hydra.utils.instantiate(config)
    inspector.run()
    inspector.save()
```

See complete example here. -->

<!-- ## Citing -->
