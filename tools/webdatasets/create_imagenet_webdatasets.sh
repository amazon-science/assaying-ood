#!/bin/bash

create_webdatset_and_copy_to_s3() {
  # Convert bash arrays to adequately formatted tuple-strings
  split_ratios=\($(echo ${SPLIT_RATIOS[*]} | sed -E 's/ /, /g'),\)  # eg. (.8 .2) -> '(.8, .2,)'
  split_names=\(\"$(echo ${SPLIT_NAMES[*]} | sed -E 's/ /", "/g')\",\)  # eg. (n1, n2) -> '("n1", "n2",)'

  echo "Generating ${NAME_TAG} with split-ratios=${split_ratios} and split-names=${split_names}..."
  python3 create_webdataset.py \
    "${SRC_DIR}" \
    "${TARGET_DIR}/" \
    --split-ratios "$split_ratios" \
    --split-names "$split_names" \
    --shuffle

  echo "Updating csv file and copying dataset to s3 bucket..."
  for split_name in ${SPLIT_NAMES[*]}
  do
    number_of_shards=$(find ${TARGET_DIR}/${split_name}/*.tar | wc -l)
    if [ $number_of_shards == 1 ]
    then
      shards="${S3LOCATION}/${split_name}/shard-000000.tar"
    else
      ((max_shard_index=${number_of_shards}-1))
      shards="${S3LOCATION}/${split_name}/shard-{000000..$(printf "%06d" ${max_shard_index})}.tar"
    fi
    number_of_examples=$(for f in $(find ${TARGET_DIR}/${split_name}/*.tar); do tar tf $f | grep 'jpg\|png'; done | wc -l)
    number_of_classes=$(find ${SRC_DIR}/ -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ $split_name == "all" ]
    then
      text_for_csv=\"${NAME_TAG}\",\"${shards}\",${number_of_examples},${number_of_classes}
    else
      text_for_csv=\"${NAME_TAG}-${split_name}\",\"${shards}\",${number_of_examples},${number_of_classes}
    fi
    echo ${text_for_csv}
    echo ${text_for_csv} >> s3_webdatasets.csv
    aws s3 cp ${TARGET_DIR}/${split_name} ${S3LOCATION}/${split_name} --recursive
  done
}

TARGET_DIR=/efs/webdata/ImageNet1k/train
SRC_DIR=/efs/ImageNet_1k/train
S3LOCATION=s3://inspector-data/sharded/ImageNet1k/train
NAME_TAG=S3ImageNet1k-train
SPLIT_RATIOS=(1.)
SPLIT_NAMES=("all")
create_webdatset_and_copy_to_s3

TARGET_DIR=/efs/webdata/ImageNet1k/val
SRC_DIR=/efs/ImageNet_1k/val
S3LOCATION=s3://inspector-data/sharded/ImageNet1k/val
NAME_TAG=S3ImageNet1k-val
SPLIT_RATIOS=(1.)
SPLIT_NAMES=("all")
create_webdatset_and_copy_to_s3

SPLIT_RATIOS=(.5 .5)
SPLIT_NAMES=("subset-test" "subset-val")
create_webdatset_and_copy_to_s3

TARGET_DIR=/efs/webdata/ImageNetA
SRC_DIR=/efs/ImageNet_A
S3LOCATION=s3://inspector-data/sharded/ImageNetA
NAME_TAG=S3ImageNetA
SPLIT_RATIOS=(1.)
SPLIT_NAMES=("all")
create_webdatset_and_copy_to_s3

SPLIT_RATIOS=(.8 .1 .1)
SPLIT_NAMES=("train" "test" "val")
create_webdatset_and_copy_to_s3

TARGET_DIR=/efs/webdata/ImageNetO
SRC_DIR=/efs/ImageNet_O
S3LOCATION=s3://inspector-data/sharded/ImageNetO
NAME_TAG=S3ImageNetO
SPLIT_RATIOS=(1.)
SPLIT_NAMES=("all")
create_webdatset_and_copy_to_s3

SPLIT_RATIOS=(.8 .1 .1)
SPLIT_NAMES=("train" "test" "val")
create_webdatset_and_copy_to_s3
