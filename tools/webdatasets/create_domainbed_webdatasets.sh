#!/bin/bash

create_webdatset_and_copy_to_s3() {
  # Convert bash arrays to adequately formatted tuple-strings
  split_ratios=\($(echo ${SPLIT_RATIOS[*]} | sed -E 's/ /, /g'),\)  # eg. (.8 .2) -> '(.8, .2,)'
  split_names=\(\"$(echo ${SPLIT_NAMES[*]} | sed -E 's/ /", "/g')\",\)  # eg. (n1, n2) -> '("n1", "n2",)'

  for dataset in ${DATASETS}
  do
    echo "Generating ${NAME_TAG}-${dataset} "\
         "with split-ratios=${split_ratios} and split-names=${split_names}..."
    python3 create_webdataset.py \
      "${SRC_DIR}/${dataset}" \
      "${TARGET_DIR}/${dataset}/" \
      --split-ratios "$split_ratios" \
      --split-names "$split_names" \
      --shuffle \
      --recursive
  done

  echo "Updating csv file and copying datasets to s3 bucket..."
  for dataset in ${DATASETS}
  do
    for environment in $(find ${TARGET_DIR}/${dataset} -mindepth 1 -maxdepth 1 -type d)
    do
      environment_name=$(basename ${environment})
      for split_name in ${SPLIT_NAMES[*]}
      do
        subfolder=${dataset}/${environment_name}/${split_name}
        number_of_shards=$(find ${TARGET_DIR}/${subfolder}/*.tar | wc -l)
        if [ $number_of_shards == 1 ]
        then
          shards="${S3LOCATION}/${subfolder}/shard-000000.tar"
        else
          ((max_shard_index=${number_of_shards}-1))
          shards="${S3LOCATION}/${subfolder}/shard-{000000..$(printf "%06d" ${max_shard_index})}.tar"
        fi
        number_of_examples=$(for f in $(find ${TARGET_DIR}/${subfolder}/*.tar); do tar tf $f | grep 'jpg\|png'; done | wc -l)
        number_of_classes=$(find ${SRC_DIR}/${dataset}/${environment_name}/ -mindepth 1 -maxdepth 1 -type d | wc -l)
        if [ $split_name == "all" ]
        then
          text_for_csv=\"${NAME_TAG}-${dataset}-${environment_name}\",\"${shards}\",${number_of_examples},${number_of_classes}
        else
          text_for_csv=\"${NAME_TAG}-${dataset}-${environment_name}-${split_name}\",\"${shards}\",${number_of_examples},${number_of_classes}
        fi
        echo ${text_for_csv}
        echo ${text_for_csv} >> s3_webdatasets.csv
        aws s3 cp ${TARGET_DIR}/${subfolder} ${S3LOCATION}/${subfolder} --recursive
      done
    done
  done
}

# Leaving out: camelyon17_v1.0 MNIST fmow_v1.1
DATASETS="VLCS PACS domain_net domain_net_subset office_home sviro terra_incognita"

TARGET_DIR=/efs/webdata/DomainBed
SRC_DIR=/efs/domainbed/data
S3LOCATION=s3://inspector-data/sharded/DomainBed
NAME_TAG=S3DomainBed
SPLIT_RATIOS=(1.)
SPLIT_NAMES=("all")
create_webdatset_and_copy_to_s3

SPLIT_RATIOS=(.8 .1 .1)
SPLIT_NAMES=("train" "test" "val")
create_webdatset_and_copy_to_s3

# Remark: Latest domainbed webdataset version to date (Sept. 29 2022) was built using
# DATASETS="VLCS PACS office_home sviro domain_net terra_incognita camelyon17 fmow"
# SRC_DIR=/efs/domainbed2/data
