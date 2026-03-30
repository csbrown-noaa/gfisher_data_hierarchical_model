from hierarchical_yolo.hierarchical_detection import HierarchicalDetectionTrainer
from hierarchical_yolo.yolo_utils import get_yolo_class_names
import yaml
from importlib import resources
from hierarchical_loss.hierarchy import Hierarchy
import ultralytics
import json
import os

#with open(HIERARCHY, 'r') as f:
#    GFISHER_HIERARCHY = json.load(f)
GFISHER_HIERARCHY_JSON = resources.files('gfisher_data_hierarchical_model.models').joinpath('hierarchy.json')
with open(GFISHER_HIERARCHY_JSON, 'r') as f:
    GFISHER_HIERARCHY = json.load(f)

YOLO_DATASET_YAML = resources.files('gfisher_data_hierarchical_model.models').joinpath('hierarchical_gfisher.yaml')
with open(YOLO_DATASET_YAML, 'r') as f:
    COCO_YOLO_ID_MAP = get_yolo_class_names(f)

class GFISHERHierarchicalDetectionTrainer(HierarchicalDetectionTrainer):
    # Hierarchy requires the index -> name map in the other direction
    _hierarchy = Hierarchy(GFISHER_HIERARCHY, {v: k for k,v in COCO_YOLO_ID_MAP.items()})
