from ood_inspector.datasets.dataset import ChainedDataset, InspectorDataset, TransformationStack
from ood_inspector.datasets.webdataset import get_webdataset

# Register ImageNet input statistics
IMAGENET_INPUT_SIZE = (3, 224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
