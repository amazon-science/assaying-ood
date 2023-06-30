#!/bin/bash

### Runs an example evaluation of a couple of models for multiple metrics.

# To be safe, switch into the folder that contains this script.
cd "$( cd "$( dirname "$0" )" && pwd )" || exit

MODELS=[\
timm_pretrained_resnet50d\
timm_pretrained_swin_small_patch4_window7_224\
]


METRICS=[\
classification_accuracy,\
demographic_parity_inferred_groups,\
expected_calibration_error,\
negative_log_likelihood,\
number_of_parameters\
]


env PYTHONPATH=../src python3 ../bin/run.py -m \
    +model=$MODEL \
    +evaluations="$METRICS" \
    ++dataloader.batch_size=64 \
    +datasets=S3ImageNet1k-val
