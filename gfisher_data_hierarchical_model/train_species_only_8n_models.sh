#!/bin/bash

# ==========================================
# SHARED CONFIGURATION
# ==========================================
WORKSPACE="~/datasets/gfisher_species_only_hierarchical_workspace"
MODEL_DIR="~/Models/hierarchical_yolo_species_only_models"
VERSION="1.0.0"
EPOCHS=80
BATCH=8
WORKERS=0
BASE_MODEL="yolov8n.pt"

echo "=========================================="
echo "🚀 Starting YOLOv8n Species-Only Training"
echo "=========================================="

# ==========================================
# 1. YOLOv8n Flat (Depth 000)
# ==========================================
echo -e "\n---> Training Flat YOLOv8n Baseline (Depth 000)"
# The combination of 'project' and 'name' guarantees the model saves to:
# ~/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_flat_level_000/1.0.0/weights/best.pt
yolo train \
    data=$WORKSPACE/tier_yolo_flat_specialists/000/train.yaml \
    model=$BASE_MODEL \
    project=$MODEL_DIR/ultralytics_yolov8n_flat_level_000 \
    name=$VERSION \
    epochs=$EPOCHS \
    workers=$WORKERS \
    batch=$BATCH

# ==========================================
# 2. YOLOv8n Flat (Depth 017 - Max Depth)
# ==========================================
echo -e "\n---> Training Flat YOLOv8n Baseline (Depth 017)"
# The combination of 'project' and 'name' guarantees the model saves to:
# ~/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt
yolo train \
    data=$WORKSPACE/tier_yolo_flat_specialists/017/train.yaml \
    model=$BASE_MODEL \
    project=$MODEL_DIR/ultralytics_yolov8n_flat_level_leaf \
    name=$VERSION \
    epochs=$EPOCHS \
    workers=$WORKERS \
    batch=$BATCH

# ==========================================
# 3. YOLOv8n Hierarchical
# ==========================================
echo -e "\n---> Training Hierarchical YOLOv8n"
# By passing the project_name as the model_name/version, it stays organized.
# Note: Because train.py is a curriculum trainer, it will append a folder for the depth.
# Your final weights will likely end up at:
# ~/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_hierarchical/1.0.0/curriculum_depth_017/weights/best.pt
python -m hierarchical_yolo.train \
    --workspace_dir $WORKSPACE \
    --model_dir $MODEL_DIR \
    --project_name ultralytics_yolov8n_hierarchical/$VERSION \
    --base_model $BASE_MODEL \
    --final_epochs $EPOCHS \
    --workers $WORKERS \
    --batch $BATCH

echo -e "\n✅ All YOLOv8n training tasks dispatched!"
