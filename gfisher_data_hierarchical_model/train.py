from ultralytics import YOLO
import os
from gfisher_data_hierarchical_model.gfisher_hierarchical_model import GFISHERHierarchicalDetectionTrainer
import hierarchical_yolo.yolo_utils
import hierarchical_loss.path_utils
import hierarchical_loss.viz_utils
import ultralytics
import torch
from PIL import Image
import time

import random

DATASETS = os.path.expanduser('~/datasets')
WEIGHTS = os.path.expanduser('~/Models/weights')
RUNS = os.path.expanduser('~/Models/runs')

YOLO_BASE_WEIGHTS = os.path.join(WEIGHTS, "yolov8n.pt")

# Configure project runs save locations.
BASE_PROJECT = os.path.join(RUNS, 'gfisher')
HIERARCHICAL_PROJECT = os.path.join(RUNS, 'hierarchical_gfisher')

# Find where data is downloaded/stored.  This is where ultralytics will download the coco128 data.
DATA = os.path.join(DATASETS, 'gfisher')
IMAGES_PATH = os.path.join(DATA, 'hierarchical_annotations', 'images')

# Find model configurations.  These are in the repository https://github.com/csbrown-noaa/hierarchical_yolo.
# If you aren't running this from the cloned repo, you will need to go acquire these and change MODEL_CONFIGS to reflect the location of the model config files.
MODEL_CONFIGS = '../gfisher_data_hierarchical_model/models'
YOLO_MODEL_YAML = os.path.join(MODEL_CONFIGS, 'hierarchicalgfisheryolov8.yaml')
YOLO_DATASET_YAML = os.path.join(MODEL_CONFIGS, 'hierarchical_gfisher.yaml')

BATCH_FACTOR = 28 # make this as large as possible to maximize GPU memory utilization

devices = list(range(torch.cuda.device_count()))
devices = devices[1:] if devices else 'cpu'


base_model_weights = YOLO_BASE_WEIGHTS
hierarchical_model = YOLO(YOLO_MODEL_YAML).load(base_model_weights)  # build a new model from scratch


results = hierarchical_model.train(
    model=base_model_weights,
    data=YOLO_DATASET_YAML,
    project=HIERARCHICAL_PROJECT,
    epochs=30,
    imgsz=640, 
    box=20, #upweight the box loss.  The marginal confidences tend to be lower for the hierarchical case, which makes the cls_loss higher, on average.
    trainer=GFISHERHierarchicalDetectionTrainer,
    device=devices[1:],
    val=False,
    batch=len(devices[1:])*BATCH_FACTOR,
    workers=0
)
