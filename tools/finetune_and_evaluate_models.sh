#!/bin/bash

### Runs an example evaluation of a couple of models for multiple metrics.

# To be safe, switch into the folder that contains this script.
cd "$( cd "$( dirname "$0" )" && pwd )" || exit

MODELS=[\
timm_pretrained_resnet50d,\
timm_pretrained_swin_small_patch4_window7_224\
]


METRICS=[\
classification_accuracy,\
demographic_parity_inferred_groups,\
expected_calibration_error,\
negative_log_likelihood,\
number_of_parameters\
]

TEST_DATASETS=[\
S3DomainBed-VLCS-Caltech101,\
S3DomainBed-VLCS-SUN09,\
S3DomainBed-VLCS-VOC2007\
]

env PYTHONPATH=../src python3 ../bin/run.py -m \
    +model=$MODELS \
    +evaluations="$METRICS" \
    ++dataloader.batch_size=64 \
    +datasets=$TEST_DATASETS \
    adaptation=finetune \
    +adaptation.dataset=S3DomainBed-VLCS-Caltech101 \
    ++adaptation.dataloader.batch_size=64 \
    ++adaptation.optimizer.defaults.lr=0.1 \
    ++adaptation.optimizer.defaults.momentum=0.9 \
    ++adaptation.finetune_only_head=True \
    ++adaptation.number_of_epochs=10 \
    corruption=imagenet_c_type \
    ++corruption.corruption_types=["brightness"] \
    ++corruption.corruption_severities=[1,2]
