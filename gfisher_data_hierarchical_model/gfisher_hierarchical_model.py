from hierarchical_yolo.hierarchical_detection import HierarchicalDetectionTrainer
from hierarchical_yolo.yolo_utils import get_yolo_class_names
import yaml
from importlib import resources
from hierarchical_loss.hierarchy import Hierarchy
import ultralytics
import json
import os

DATASETS = ultralytics.settings['datasets_dir']
DATA = os.path.join(DATASETS, 'gfisher')
HIERARCHY_DATA = os.path.join(DATA, 'hierarchy_data')
HIERARCHY = os.path.join(HIERARCHY_DATA, 'hierarchy.json')

with open(HIERARCHY, 'r') as f:
    GFISHER_HIERARCHY = json.load(f)

YOLO_DATASET_YAML = resources.files('gfisher_data_hierarchical_model.models').joinpath('hierarchical_gfisher.yaml')
with open(YOLO_DATASET_YAML, 'r') as f:
    COCO_YOLO_ID_MAP = get_yolo_class_names(f)

class GFISHERHierarchicalDetectionTrainer(HierarchicalDetectionTrainer):
    # Hierarchy requires the index -> name map in the other direction
    _hierarchy = Hierarchy(GFISHER_HIERARCHY, {v: k for k,v in COCO_YOLO_ID_MAP.items()})
