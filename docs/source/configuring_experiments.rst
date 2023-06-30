Configuring an experiment
=========================

``inspector`` is meant to quick-start experimentation when training neural networks.

The framework orchestrates the entire data pipeline, but allows you to steer the runtime behaviour
through a config-driven API based on
`StructuredConfigs from the Hydra package <https://hydra.cc/docs/tutorials/structured_config/intro>`_.


.. The API of the framework is config-driven, based on
.. `StructuredConfigs from the Hydra package <https://hydra.cc/docs/tutorials/structured_config/intro>`_.

You can specify models, datasets, evaluations and all the other supported modules and hyperparameters
only by adding or altering the respective configurations. This section describes the configurations
that you can add either in form of a yaml-file, or via command-line arguments.

Command-line interface
----------------------

You can run ``inspector`` from the CLI and provide the experiment configuration as command line arguments.

The following command evaluates a pretrained *ResNet-18* model on the validation and test splits of
the *ImageNet* dataset, measuring the *classification accuracy* and the *negative log-likelihood*:

.. _base_cli_example:

..  code-block:: bash

    inspector +model=timm_pretrained_resnet18 \
      +datasets=[ImageNet1k-val,ImageNet1k-test] \
      +evaluations=[negative_log_likelihood,classification_accuracy]

Configuration with yaml file
----------------------------

Equivalently, you can pass a yaml config file to describe the experiment you want to run:

..  code-block:: yaml

    defaults:
      - inspector
      - model: timm_pretrained_resnet18
      - datasets:
        - ImageNet1k-val
        - ImageNet1k-test
      - evaluations:
        - classification_accuracy
        - negative_log_likelihood

You only need to provide the location of the config file by running the following command:

..  code-block:: bash

    inspector --config-dir <config_dir> --config-name <config_name>


Inspector modules
-----------------

In this section, you can find the synthax for configuring all the base inspector modules.

.. |br| raw:: html

  <br/>

Models
......

**CLI**

To specify a model from the CLI, use the flag :code:`+model=<registered_model_name>`.

..  code-block:: bash

    inspector +model=timm_pretrained_resnet18 <additional_parameteres>

**YAML**

..  code-block:: yaml

    defaults:
      - inspector
      - model: torchvision_pretrained_resnet18


Datasets
........

You can specify *multiple* datasets to be used for evaluating the model using the ``datasets`` config.

**CLI**

To specify a list of datasets for evaluation from the CLI, use the flag: |br|
``+datasets=[<registered_dataset1>,<registered_dataset2>,<etc>]``. For a single dataset, you can omit the parantheses.

..  code-block:: bash

    inspector +datasets=[DomainBed-PACS-sketch,DomainBed-PACS-photo] <additional_parameteres>

**YAML**

..  code-block:: yaml

    defaults:
      - inspector
      - datasets:
        - DomainBed-PACS-sketch
        - DomainBed-PACS-photo


Corruptions
...........

You can add corruptions to the evaluation datasets using the ``corruption`` config. By default, it is
set to ``no_corruption``. You can override it to ``imagenet_c_type`` and provide a list of corruptions
to be applied using the ``corruption.corruption_types`` config and control the hyperparameters
``corruption.corruption_severities`` and ``corruption.combine_corruption_types``. Specify the datasets
you want to apply corruptions on using the ``corruption.datasets`` config. The corrupted datasets will
be added to the uncorrupted evaluation datasets previously defined under ``datasets``.

**CLI**

..  code-block:: bash

    inspector corruptions=imagenet_c_type \
      corruption.corruption_types=[gaussian_noise,brightness] \
      corruption.corruption_severities=[1,2] \
      corruption.combine_corruption_types=True \
      corruption.datasets=[DomainBed-PACS-sketch,DomainBed-PACS-photo]

**YAML**

..  code-block:: yaml

    defaults:
      - inspector
      - override corruption: imagenet_c_type
      - corruption.datasets:
        - DomainBed-PACS-sketch
        - DomainBed-PACS-photo

    corruption:
      corruption_types:
        - "gaussian_noise"
        - "brightness"
      corruption_severities:
        - 1
        - 2
      combine_corruption_types: True

See a detailed description of the corruption parameters in the `corruption API documentation <link docu>`_.

Evaluation metrics
..................

**CLI**

To specify a list of evaluation metrics from the CLI, use the flag: |br|
``+evaluations=[<registered_evaluation1>,<registered_evaluation2>,<etc>]``. For a single metric, you can omit the parantheses.

..  code-block:: bash

    inspector +evaluations=[classification_accuracy,negative_log_likelihood] <additional_parameteres>

**YAML**

..  code-block:: yaml

    defaults:
      - inspector
      - evaluations:
        - classification_accuracy
        - negative_log_likelihood


Adaptation strategies
.....................

By default, the ``no_adaptation`` configuration is selected. This means that you only need to select
a model, an evaluation dataset and a metric and you can already run an inspector experiment that
evaluates the model without performing any adaptation, as in our first :ref:`example <base_cli_example>`.

To finetune the model, you need to override the ``adaptation`` config and set it to ``finetune``.
  - You will need to provide a dataset to train on under the ``adaptation.dataset`` config.
  - You can also add augmentations to be applied on the adaptation dataset under the ``adaptation.dataset.transformations.augmenter`` config.
    By default, no augmentation is applied.
  - A sequence of base transformations like resizing and normalization is applied by default
    and can be overriden using the ``adaptation.dataset.transformations.transformation`` config.
  - You can finetune the whole model or only the head by setting ``adaptation.finetune_only_head``.
    The default behaviour is to train the whole model.
  - Set the number of epochs using the ``adaptation.number_of_epochs`` config.
  - Set the optimizer name in ``adaptation.optimizer.classname`` and pass any constructor arguments
    in ``adaptation.optimizer.defaults``. All `torch.optim` optimizers are supported.
  - You can use a learning rate scheduler by modifying the ``adaptation.lr_scheduler`` config. By
    default it is set to ``None``. Override it to ``torch`` and then set the classname and constructor
    arguments in ``adaptation.lr_scheduler.classname`` and ``adaptation.lr_scheduler.options``, respectively.

You can see a complete adaptation configuration bellow:

**CLI**

..  code-block:: bash

    inspector adaptation=finetune \
      adaptation.dataset=DomainBed-PACS-sketch \
      adaptation.number_of_epochs=5 \
      +adaptation.finetune_only_head=True \
      +adaptation.optimizer.classname=SGD \
      +adaptation.optimizer.defaults.lr=0.001 \
      adaptation.lr_scheduler=torch \
      +adaptation.lr_scheduler.classname=MultiStepLR \
      +adaptation.lr_scheduler.options.milestones=[1,2,3] \
      <additional_parameteres>

**YAML**

..  code-block:: yaml

    defaults:
      - inspector
      - override adaptation: finetune
      - override adaptation/lr_scheduler: torch
      - adaptation.dataset: DomainBed-PACS-sketch

    adaptation:
      number_of_epochs: 5
      finetune_only_head: True
      optimizer:
        classname: SGD
        defaults:
          lr: 0.001
      lr_scheduler:
        classname: MultiStepLR
        options:
          milestones: [1, 2, 3]

.. note::
    It is important to note that the ``defaults`` list is the only place where internal config names
    such as ``DomainBed-PACS-sketch`` and ``finetune`` can be used. **In the config part of the
    yaml file these names would simply be interpreted as strings!** (like ``SGD`` and ``MultiStepLR``)

.. warning::
    Be aware of the distinction between evaluation datasets (registered under ``datasets``) and
    adaptation datasets (registered under ``adaptation.dataset``). They load the same raw datasets,
    but only the adaptation config supports an augmenter and the default base transformations are
    slightly different. See more details about the `transformation stack <link>`_.


Examples
--------

To wrap up, here is a complete experiment configuration using all the settings we discussed above:

You can find more yaml config examples for various experiments here: `YAML config examples <link_config_files>`_.