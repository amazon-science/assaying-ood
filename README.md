# Assaying OOD

The *Assaying Out-of-Distribution Inspector* is a library to evaluate robustness, generalization and
fairness properties of neural networks. It is highly configurable with the goal of speeding up and 
standardizing the robustness testing process of neural networks. It includes a transfer learning
classification benchmark for long-tail tasks with 40+ data sets each with multiple real
out-of-distribution (OOD) test sets as well as fairness data sets. For evaluation it supports
standard accuracy metrics, calibration error, adversarial attacks, demographic parity, explanation
infidelity, synthetic corruptions on both in- and out-of-distribution data. For fine-tuning it
supports several augmentation strategies (>10 standard transformations plus popular methods such as
auto-augment, random-augment, generalized MixUp, and AugMix).

The *Inspector* has been the basis for our
[NeurIPS'22 paper on out-of-distribution robustness](https://arxiv.org/abs/2207.09239)
If you use the *Inspector* in your research, please cite:
```plain
@inproceedings{Wenzel2022AssayingOOD,
  title={Assaying Out-Of-Distribution Generalization in Transfer Learning},
  author={Florian Wenzel and Andrea Dittadi and Peter V. Gehler and Carl-Johann Simon-Gabriel and Max Horn and Dominik Zietlow and David Kernert and Chris Russell and Thomas Brox and Bernt Schiele and Bernhard Sch\"olkopf and Francesco Locatello},
  booktitle={Neural Information Processing Systems},
  year={2022},
}
```


## Setup
- We recommend using python 3.8.
- Install [ImageMagick](https://imagemagick.org/). E.g., by
  - Linux: `sudo yum install ImageMagick-devel`
  - Mac: `brew install freetype imagemagick`
- Install required python packages by `pip install -r requirements_dev.txt`.
- If you use a Mac, additionally install `pip install -r requirements_mac.txt`.

## Dataset hosting
Please use the scripts in `tools/webdatasets` to download and prepare the datasets. In the codebase
we currently assume that all the datasets are stored in a S3 bucket under the
path `s3://inspector-data/`. If you want to store the data somewhere else, please adjust the paths 
in the csv files in `src/ood_inspector/api/datasets`.

## Quick overview on how to run the *Inspector*
If you want to quickly get an idea on how to run the inspector check out the following examples. For
a more in-depth guide see the next sections.

*How to run the Inspector from the command line?* Run (and inspect) one of the following
commands:
- `tools/evaluate_models_imagenet.sh`
- `tools/finetune_and_evaluate_models.sh`

*How to run a sweep on a single machine?* Check out the experiment configs in 
`config_files/examples/`. For instance, a small example sweep is defined in
- `config_files/examples/example_small_sweep.yaml`


## General usage
All experiments are executed using [Hydra](https://hydra.cc/). They can be either run from the
command line (CLI) or via yaml files.
### CLI

#### Datasets and Evaluations

Usually we want to be able to evaluate models with respect to certain metrics
on datasets (or splits thereof) of our choice.  The datasets on which
evaluations should be run are defined in the `datasets` dictionary of the
inspector configuration.  We can set these using overrides on the command line
or in a yaml file (see below).  For example, in order to run evaluations on the ImageNet1k dataset
we can call `run.py` with

```bash
PYTHONPATH=src python bin/run.py datasets=[S3ImageNet1k_val] <additional parameters>
```


Evaluations are defined in a similar manner, here the dictionary
`evaluations` is responsible for tracking evaluation metrics and metrics can be
added using a similar syntax as above:

```bash
PYTHONPATH=src python bin/run.py evaluations=[negative_log_likelihood,classification_accuracy] <additional parameters>
```

### Configuration with yaml files
For example experiment configs, please have a look at `config_files/examples/`.

#### Using config groups

In order to run the above CLI commands using a yaml file, we can denote the
command in the `defaults` part of the yaml file.  It is important to note that
the defaults list is the only place where internal config names such as
`negative_log_likelihood` and `S3ImageNet1k_val` can be used.  *In the config part of
the yaml file these names would simply be interpreted as strings!*

Let's look at an example which combines the CLI calls presented in the "Datasets
and Evaluations" section with some additional parameters:

```yaml
default:
  - datasets: S3ImageNet1k_val
  - evaluations: [negative_log_likelihood, classification_accuracy]
```

#### Advanced configuration

While config groups are nice for convenience, they are limited in their scope
and thus flexibility.  If we for example want to compute top_k accuracy for
different top_k values we need to be able to denote that somehow.  For this we
store all config classes in a separate config group called `schemas`.  We can
use `schemas` anywhere where we would want to access the configuration class
itself.

An example of a more advanced configuration
```yaml
default:
  - schemas/evaluations/classification_accuracy@evaluations.top5_acc
  - schemas/evaluations/classification_accuracy@evaluations.top10_acc
  - _self_

evaluations:
  top5_acc:
    top_k: 5
  top10_acc:
    top_k: 10
```

We see that the schema entry in the defaults list defines which *object type*
should be associated with an entry in the `evaluations` dict, whereas the
entries in the config section determine the parameters that should be set of
those objects.

Examples for using the *Inspector* with yaml files can be found in the path
`config_files/examples`.

Below you can find a list of example config names with a short description:

 * **advanced_metrics:** Shows how you can construct dictionaries of metrics
   that are fully customizable to your needs.


The *Inspector* can then be called on these using the command
`PYTHONPATH=src python bin/run.py --config-dir config_files/examples --config-name <config_name>`.

## Run the *Inspector* on a cluster
The *Inspector* can be easily launched on a cluster leveraging hydra's launcher plugin. For
instance, it can be run on a Ray cluster or a SLURM cluster. For more information, check out the
[Hydra](https://hydra.cc/) docs.

## Inspecting the results

The results are stored in `results.json`.
