import copy
from typing import Callable, Tuple

import webdataset


def search_insertation_points(
    root: webdataset.Composable, criterion: Callable[[webdataset.Composable], bool]
) -> Tuple[webdataset.Composable, webdataset.Composable]:
    """Find the insertion point in a linked list of webdataset.Processors.

    This function returns the node which first triggers `criterion` to become true and the node
    which was found prior to it.  This allows us to insert nodes between two similar to inserting
    nodes in a linked list.
    """
    previous_node = None
    cur_node = root

    while hasattr(cur_node, "source") or hasattr(cur_node, "dataset"):
        child = "source" if hasattr(cur_node, "source") else "dataset"
        if criterion(cur_node):
            return cur_node, previous_node
        else:
            previous_node = cur_node
            cur_node = getattr(cur_node, child)
    # Edge case, last node in list satisfied criterion.
    if criterion(cur_node):
        return cur_node, previous_node
    raise RuntimeError("Cannot find node which satisfies criterion.")


def pytorch_shard_list_criterion(obj: webdataset.Composable):
    return isinstance(obj, webdataset.PytorchShardList)


def shuffle_webdataset_pipeline(
    dataset: webdataset.Processor, shuffle_buffer_size: int, epoch_shuffle=False
):
    # Create copy of data processing pipeline to avoid unexpected effects.
    dataset = copy.deepcopy(dataset)

    # Activate shard list shuffling.
    shard_list: webdataset.PytorchShardList
    shard_list, _ = search_insertation_points(
        dataset, lambda obj: isinstance(obj, webdataset.PytorchShardList)
    )
    shard_list.shuffle = True
    shard_list.epoch_shuffle = epoch_shuffle

    # Find group_by_keys operation which is the first where instances are expanded from the tar
    # files.
    def criterion_group_by_keys(obj):
        if not hasattr(obj, "f"):
            return False
        return id(obj.f) == id(webdataset.group_by_keys)

    group_by_keys, subsequent_op = search_insertation_points(dataset, criterion_group_by_keys)
    new_chain = group_by_keys.shuffle(shuffle_buffer_size)
    # Update subsequent op to use new chain with shuffle instead.
    subsequent_op.source_(new_chain)
    return dataset
