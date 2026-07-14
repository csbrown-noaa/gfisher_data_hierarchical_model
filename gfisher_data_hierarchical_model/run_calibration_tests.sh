#!/bin/bash

# You can keep this as apples2apples_eval or separate it
EVAL_DIR="./calibration_eval"
mkdir -p $EVAL_DIR

# ==========================================
# SHARED PATHS
# ==========================================
# The Calibration benchmark requires the raw COCO format ground truth JSON 
# to perform the ecosystem-independent bipartite matching.
GT_JSON="$HOME/datasets/gfisher_workspace/master_coco/test.json"
HIERARCHY_JSON="$HOME/datasets/gfisher_workspace/hierarchy.json"

# Change these to the Species-Only YAMLs once they are finished
HIERARCHICAL_YAML="$HOME/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml"
FLAT_YAML="$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml"

# ==========================================
# 1. YOLOv11x Hierarchical
# ==========================================
RUN_NAME="ultralytics_yolov11x_hierarchical_calibration_benchmark"
python -m hierarchical_yolo.calibration_benchmarks pipeline \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt" \
     --model_type hierarchical \
     --data_yaml $HIERARCHICAL_YAML \
     --gt_json $GT_JSON \
     --hierarchy_json $HIERARCHY_JSON \
     --flat_baseline_yaml $FLAT_YAML \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 2. YOLOv11x Flat (Leaf)
# ==========================================
RUN_NAME="ultralytics_yolov11x_flat_level_leaf_calibration_benchmark"
python -m hierarchical_yolo.calibration_benchmarks pipeline \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
     --model_type flat \
     --data_yaml $FLAT_YAML \
     --gt_json $GT_JSON \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 3. YOLOv8n Hierarchical
# ==========================================
RUN_NAME="ultralytics_yolov8n_hierarchical_calibration_benchmark"
python -m hierarchical_yolo.calibration_benchmarks pipeline \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt" \
     --model_type hierarchical \
     --data_yaml $HIERARCHICAL_YAML \
     --gt_json $GT_JSON \
     --hierarchy_json $HIERARCHY_JSON \
     --flat_baseline_yaml $FLAT_YAML \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 4. YOLOv8n Flat (Leaf)
# ==========================================
RUN_NAME="ultralytics_yolov8n_flat_level_leaf_calibration_benchmark"
python -m hierarchical_yolo.calibration_benchmarks pipeline \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
     --model_type flat \
     --data_yaml $FLAT_YAML \
     --gt_json $GT_JSON \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log
