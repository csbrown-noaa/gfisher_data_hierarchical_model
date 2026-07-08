#!/bin/bash

EVAL_DIR="./apples2apples_eval"
mkdir -p $EVAL_DIR

# ==========================================
# 1. YOLOv11x Flat (Level Leaf to Objectness)
# ==========================================
RUN_NAME="ultralytics_yolov11x_flat_level_leaf_objectness_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt \
     --model_type flat \
     --data_yaml ~/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 2. YOLOv11x Flat (Level 000 to Objectness)
# ==========================================
RUN_NAME="ultralytics_yolov11x_flat_level_000_objectness_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_000/1.0.0/weights/best.pt \
     --model_type flat \
     --data_yaml ~/datasets/gfisher_workspace/tier_yolo_flat_specialists/000/train.yaml \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 3. YOLOv11x Hierarchical (to Objectness)
# ==========================================
RUN_NAME="ultralytics_yolov11x_hierarchical_objectness_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt \
     --model_type hierarchical \
     --data_yaml ~/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml \
     --hierarchy_json ~/datasets/gfisher_workspace/hierarchy.json \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 4. YOLOv8n Hierarchical (to Objectness)
# ==========================================
RUN_NAME="ultralytics_yolov8n_hierarchical_objectness_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt \
     --model_type hierarchical \
     --data_yaml ~/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml \
     --hierarchy_json ~/datasets/gfisher_workspace/hierarchy.json \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 5. YOLOv8n Flat (Level 000 to Objectness)
# ==========================================
RUN_NAME="ultralytics_yolov8n_flat_level_000_objectness_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_000/1.0.0/weights/best.pt \
     --model_type flat \
     --data_yaml ~/datasets/gfisher_workspace/tier_yolo_flat_specialists/000/train.yaml \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log

# ==========================================
# 6. YOLOv8n Flat (Level 017 to Objectness)
# ==========================================
RUN_NAME="ultralytics_yolov8n_flat_level_017_objectness_benchmark"
python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights ~/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt \
     --model_type flat \
     --data_yaml ~/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml \
     --split test \
     --project $EVAL_DIR \
     --name $RUN_NAME > $EVAL_DIR/${RUN_NAME}.log
