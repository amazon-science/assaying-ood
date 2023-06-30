# The following datasets were copied from the old s3 bucket to the new one with appropriate updates in the .csv file.
# aws s3 cp s3://inspector-datasets/sharded/ImageNet_C_all_1 s3://inspector-data/sharded/ImageNetC/AllCorruptionsOfSize1/all --recursive
# aws s3 cp s3://inspector-datasets/sharded/ImageNet_C_all_2 s3://inspector-data/sharded/ImageNetC/AllCorruptionsOfSize2/all --recursive
# aws s3 cp s3://inspector-datasets/sharded/ImageNet_C_all_3 s3://inspector-data/sharded/ImageNetC/AllCorruptionsOfSize3/all --recursive
# aws s3 cp s3://inspector-datasets/sharded/ImageNet_C_all_4 s3://inspector-data/sharded/ImageNetC/AllCorruptionsOfSize4/all --recursive
# aws s3 cp s3://inspector-datasets/sharded/ImageNet_C_all_5 s3://inspector-data/sharded/ImageNetC/AllCorruptionsOfSize5/all --recursive
# aws s3 cp s3://inspector-datasets/sharded/TinyImageNetForTesting s3://inspector-data/sharded/TinyImageNetForTesting --recursive

# All other imagenet-type datasets and the domainbed datasets were re-created for the new s3 bucket
./create_imagenet_webdatasets.sh
./create_domainbed_webdatasets.sh
sort -o s3_webdatasets.csv s3_webdatasets.csv
