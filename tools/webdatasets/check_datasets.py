import os

import webdataset as wds

"""
This script runs various safety checks on the domainbed webdatasets.

For every domainbed dataset and environment, the script checks that the
train-val-test splits do not overlap, that their union is equal to the total
dataset (i.e., is equal to the split called 'all'), and whether the new version
of the domainbed webdatasets (downloaded and sharded on Jan 31 - Feb 1 2022)
differ from the old ones we had. Those tests all rely on the '_filename' key
stored with every image in the webdataset.

The script generates a dataset_checks_results.txt file, which could look as follows:

    VLCS            DomainBed       LabelMe         train-val    True
    VLCS            DomainBed       LabelMe         train-test   True
    VLCS            DomainBed       LabelMe         val-test     True
    VLCS            DomainBed       LabelMe         all==union   True
    VLCS            DomainBedOld    LabelMe         train-val    True
    VLCS            DomainBedOld    LabelMe         train-test   True
    VLCS            DomainBedOld    LabelMe         val-test     True
    VLCS            DomainBedOld    LabelMe         all==union   True
    VLCS            DomainBed+Old   LabelMe         new==old     True

    VLCS            DomainBed       Caltech101      train-val    True
    VLCS            DomainBed       Caltech101      train-test   True
    ....

The keyword `True` indicates a successful test. Here, all tests were
successfull: no overlap between splits (train-val, train-test and val-test),
union of splits are equal to the to the total dataset (all==union), and old and
new datasets contain the same images (new==old).
"""


def check_intersections(fn):
    """Checks that the train-val-test splits do not overlap."""
    splits = ["train", "val", "test"]
    output = []
    fn = fn[dir_name]
    for i in range(3):
        s1 = splits[i]
        for j in range(i + 1, 3):
            s2 = splits[j]
            test = f"{dataset_name:<15} {dir_name:<15} {env:15} {s1}-{s2}"
            line = f"{test:<60} {len(fn[s1].intersection(fn[s2])) == 0}"
            output.append(line)
    return "\n".join(output)


def check_union_is_all(fn):
    """Checks that the union of train-val-test splits is equal to the 'all' split."""
    union_of_splits = set()
    fn = fn[dir_name]
    for split in ["train", "val", "test"]:
        union_of_splits.update(fn[split])
    test = f"{dataset_name:<15} {dir_name:<15} {env:<15} all==union"
    return f"{test:<60} {union_of_splits == fn['all']}"


def check_equality(fn, dataset_name):
    """Checks that the old version of the dataset equals the newly downloaded one."""
    test = f"{dataset_name:<15} {'DomainBed+Old':<15} {env:<15} new==old"
    return f"{test:<60} {fn['DomainBed']['all'] == fn['DomainBedOld']['all']}"


if __name__ == "__main__":
    datasets = [
        "VLCS",
        "PACS",
        "office_home",
        "sviro",
        "terra_incognita",
        "domain_net",
        # "domain_net_subset",  # Was not re-generated: we still use the old one.
    ]
    with open("check_datasets_results.txt", "a") as file:
        for dataset_name in datasets:
            fn = dict()
            basepath = f"/efs/webdata/DomainBed/{dataset_name}"
            envs = [
                env for env in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, env))
            ]
            for i, env in enumerate(envs):
                for dir_name in ["DomainBed", "DomainBedOld"]:
                    fn[dir_name] = dict()
                    ds = os.path.join("/efs/webdata/", dir_name, dataset_name)
                    for split in ["all", "train", "val", "test"]:
                        fn[dir_name][split] = set()
                        tars = [
                            fname
                            for fname in os.listdir(os.path.join(ds, env, split))
                            if fname.endswith(".tar")
                        ]
                        for tarfile in tars:
                            uri = os.path.join(ds, env, split, tarfile)
                            loader = wds.WebDataset(uri)
                            for sample in loader:
                                for key, value in sample.items():
                                    if key == "_filename":
                                        fn[dir_name][split].add(repr(value))
                    print(check_intersections(fn))
                    print(check_union_is_all(fn))
                    file.write(check_intersections(fn) + "\n")
                    file.write(check_union_is_all(fn) + "\n")
                print(check_equality(fn, dataset_name) + "\n")
                file.write(check_equality(fn, dataset_name) + "\n\n")
