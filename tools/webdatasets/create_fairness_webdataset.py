"""
Turn a local fairness dataset into a sharded webdataset.

This script assumes that the dataset comes with a folder of images and a file comprising various
attributes for every image.
"""

import argparse
import json
import os
import pandas
import webdataset


### Set parameters for ShardWriter and various paths
maxcount=10000
maxsize=int(3e8)


parser = argparse.ArgumentParser(description="Create sharded fairness webdataset.")
parser.add_argument("image_folder", type=str, help="path to folder with images")
parser.add_argument("file_with_attributes", type=str, help="path to csv-file comprising attributes "
                                        "for images -- first column has to provide image names and "
                                        "first row has to provide attribute names")
parser.add_argument("output_folder", type=str, help="path to output folder")
parser.add_argument("--row-label-prefix", dest="prefix", type=str, default="", help="prefix to "
                                    "image name in first column of file_with_attributes (e.g., "
                                    "'train/' or 'val/' for FairFace; default: "")")
args = parser.parse_args()
path_to_images = args.image_folder
path_to_attributes_file = args.file_with_attributes
path_to_output_folder = args.output_folder
index_prefix = args.prefix

if not path_to_images[-1]=="/":
    path_to_images = path_to_images + "/"
if not path_to_output_folder[-1]=="/":
    path_to_output_folder = path_to_output_folder + "/"

if not os.path.exists(path_to_images):
    raise ValueError("Path to folder with images '"+path_to_images+"' does not exist")
if not os.path.exists(path_to_output_folder):
    raise ValueError("Path to output folder '"+path_to_output_folder+"' does not exist")
if not os.path.exists(path_to_attributes_file):
    raise ValueError("Path to file with attributes '"+path_to_attributes_file+"' does not exist")


attributes=pandas.read_csv(path_to_attributes_file,header=0,index_col=0)
attributes=attributes.astype('str')


attribute_to_integer_dict={}
for attribute_name in attributes.columns:
    attribute_to_integer_dict[attribute_name]={}
    for counter,attr in enumerate(sorted(pandas.unique(attributes[attribute_name]).tolist())):
        attribute_to_integer_dict[attribute_name][attr]=counter
    attributes[attribute_name]=attributes[attribute_name].map(attribute_to_integer_dict[attribute_name])
with open(path_to_output_folder+"attributes_to_integers.json", "w") as outfile:
    json.dump(attribute_to_integer_dict, outfile)


def dataset_samples():
    for img in os.listdir(path_to_images):
        if not img.endswith(".jpg"):
            continue
        img_path=path_to_images+img
        with open(img_path, "rb") as stream:
            binary_data = stream.read()
        sample={
            "__key__": img.split(".jpg")[0],
            "jpg": binary_data}
        for attribute_name in attributes.columns:
            sample[attribute_name+".cls"]=attributes[attribute_name].loc[index_prefix+img]
        yield sample


with webdataset.ShardWriter(path_to_output_folder+"shard-%06d.tar",maxsize=maxsize, maxcount=maxcount) as sink:
    for sample in dataset_samples():
        sink.write(sample)
