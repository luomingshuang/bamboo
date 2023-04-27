import torch.utils.data
from .cocodetection import CocoDetection


def get_coco_api_from_dataset(dataset):
    return dataset.coco