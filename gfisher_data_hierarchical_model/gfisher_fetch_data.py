#!/usr/bin/env python
# coding: utf-8

import ultralytics
import pycocowriter.coco2yolo
import os
import urllib.request
import json
import shutil

# Importing our newly abstracted classes
from hierarchical_loss.worms_expander import WormsCocoExpander

REFRESH = True

TRAIN_DATA_URL = "https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/train_annotations_worms.json"
TEST_DATA_URL = "https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/test_annotations_worms.json"

# Ultralytics default settings. See https://docs.ultralytics.com/quickstart/#ultralytics-settings
DATASETS = os.path.expanduser('~/datasets')

# Directories
DATA = os.path.join(DATASETS, 'gfisher')
RAW_DATA = os.path.join(DATA, 'raw_data')
HIERARCHY_DATA = os.path.join(DATA, 'hierarchy_data')
IMAGES_PATH = os.path.join(DATA, 'images')

# File Paths
TRAIN_ANNOTATIONS = os.path.join(RAW_DATA, "train_annotations.json")
TEST_ANNOTATIONS = os.path.join(RAW_DATA, "test_annotations.json")

# Output hierarchical annotations
TRAIN_HIERARCHICAL = os.path.join(DATA, "train_hierarchical_annotations.json")
VAL_HIERARCHICAL = os.path.join(DATA, "val_hierarchical_annotations.json")

# Model configs
GFISHER_MODELS = 'gfisher_data_hierarchical_model/models'
HIERARCHY_JSON = os.path.join(GFISHER_MODELS, 'hierarchy.json')
YOLO_MODEL_YAML = os.path.join(GFISHER_MODELS, 'hierarchical_gfisher_yolov8.yaml')
YOLO_DATASET_YAML = os.path.join(GFISHER_MODELS, 'hierarchical_gfisher.yaml')

# ==========================================
# 1. Setup and Get the GFISHER data
# ==========================================

for directory in [DATA, RAW_DATA, HIERARCHY_DATA]:
    if not os.path.exists(directory):
        os.mkdir(directory)

if not os.path.exists(TRAIN_ANNOTATIONS):
    urllib.request.urlretrieve(TRAIN_DATA_URL, TRAIN_ANNOTATIONS)
if not os.path.exists(TEST_ANNOTATIONS):
    urllib.request.urlretrieve(TEST_DATA_URL, TEST_ANNOTATIONS)

with open(TRAIN_ANNOTATIONS, 'r') as f:
    train_coco = json.load(f)
with open(TEST_ANNOTATIONS, 'r') as f:
    test_coco = json.load(f)

# ==========================================
# 2. Expand & Align the Categories
# ==========================================

print("Initializing Expander and fetching/building WoRMS hierarchy...")
expander = WormsCocoExpander(use_cache=True)

# Build the master tree unifying all taxa found across both datasets
expander.build_master_hierarchy(train_coco, test_coco)

# Cache original category mappings BEFORE in-place mutation occurs
# so we can mathematically assert they didn't get corrupted
def get_orig_mappings(coco_dict):
    cat_map = {cat['id']: cat['name'] for cat in coco_dict.get('categories', [])}
    return {ann['id']: cat_map[ann['category_id']] for ann in coco_dict.get('annotations', [])}

train_orig_mapping = get_orig_mappings(train_coco)
test_orig_mapping = get_orig_mappings(test_coco)

# Align the individual datasets to the new global master taxonomy
print("Aligning Training Dataset...")
aligned_train = expander.align_dataset(train_coco)

print("Aligning Testing Dataset...")
aligned_test = expander.align_dataset(test_coco)

# ==========================================
# 3. Data Quality Assertions
# ==========================================

def verify_alignment(orig_mapping: dict, new_coco: dict, split_name: str):
    """Verifies that the aligned annotations mapped to the correct taxonomic names."""
    category_map_new = {cat['id']: cat for cat in new_coco['categories']}
    
    for ann in new_coco['annotations']:
        old_cat_name = orig_mapping[ann['id']]
        new_cat_name = category_map_new[ann['category_id']]['name']
        assert old_cat_name == new_cat_name, f"Category mapping failed: {old_cat_name} != {new_cat_name}"
        
    print(f"Data quality assertions passed for {split_name} split!")

verify_alignment(train_orig_mapping, aligned_train, "Train")
verify_alignment(test_orig_mapping, aligned_test, "Val")

# ==========================================
# 4. Save Artifacts
# ==========================================

print("Saving aligned COCO files and hierarchy artifact...")
with open(TRAIN_HIERARCHICAL, 'w') as f:
    json.dump(aligned_train, f)
    
with open(VAL_HIERARCHICAL, 'w') as f:
    json.dump(aligned_test, f)

with open(HIERARCHY_JSON, 'w') as f:
    json.dump(expander.hierarchy_tree, f)

# ==========================================
# 5. Convert to YOLO
# ==========================================

if REFRESH:
    print("Converting to YOLO format...")
    pycocowriter.coco2yolo.coco2yolo(DATA, DATA)

if os.path.exists(os.path.join(DATA, 'train.yaml')):
    shutil.copyfile(os.path.join(DATA, 'train.yaml'), YOLO_DATASET_YAML)

print("Pipeline Complete.")
