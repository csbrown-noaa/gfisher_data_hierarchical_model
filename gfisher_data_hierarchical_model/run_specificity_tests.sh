#!/bin/bash

EVAL_DIR="./apples2apples_eval"
mkdir -p $EVAL_DIR

# ==========================================
# 1. YOLOv11x Hierarchical
# ==========================================
RUN_NAME="ultralytics_yolov11x_hierarchical_specificity_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks specificity \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt \
     --hierarchical_eval_yaml ~/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml \
     --flat_data_yaml ~/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml \
     --hierarchy_json ~/datasets/gfisher_workspace/hierarchy.json \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 2. YOLOv11x Flat (Leaf) - Native YOLO
# ==========================================
RUN_NAME="ultralytics_yolov11x_flat_level_leaf_specificity_benchmark"
yolo val \
     model=~/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt \
     data=~/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml \
     split=test \
     project=$EVAL_DIR \
     name=$RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 3. YOLOv8n Hierarchical
# ==========================================
RUN_NAME="ultralytics_yolov8n_hierarchical_specificity_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks specificity \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt \
     --hierarchical_eval_yaml ~/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml \
     --flat_data_yaml ~/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml \
     --hierarchy_json ~/datasets/gfisher_workspace/hierarchy.json \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 4. YOLOv8n Flat (Leaf) - Native YOLO
# ==========================================
RUN_NAME="ultralytics_yolov8n_flat_level_leaf_specificity_benchmark"
yolo val \
     model=~/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt \
     data=~/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml \
     split=test \
     project=$EVAL_DIR \
     name=$RUN_NAME > $EVAL_DIR/${RUN_NAME}.log
