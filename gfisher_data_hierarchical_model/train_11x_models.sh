#!/bin/bash

# ==========================================
# SHARED CONFIGURATION
# ==========================================
WORKSPACE="~/datasets/gfisher_workspace"
MODEL_DIR="~/Models/hierarchical_yolo_models"
VERSION="1.0.0"
FLAT_EPOCHS=80
FINAL_EPOCHS=50
SHALLOW_EPOCHS=2
BATCH=8
WORKERS=0
BASE_MODEL="yolo11x.pt"

echo "=========================================="
echo "🚀 Starting YOLOv11x Full Taxonomy Training"
echo "=========================================="

# ==========================================
# 1. YOLOv11x Flat (Depth 000)
# ==========================================
echo -e "\n---> Training Flat YOLOv11x Baseline (Depth 000)"
yolo train \
    data=$WORKSPACE/tier_yolo_flat_specialists/000/train.yaml \
    model=$BASE_MODEL \
    project=$MODEL_DIR/ultralytics_yolov11x_flat_level_000 \
    name=$VERSION \
    epochs=$FLAT_EPOCHS \
    workers=$WORKERS \
    batch=$BATCH

# ==========================================
# 2. YOLOv11x Flat (Depth 017 - Max Depth)
# ==========================================
echo -e "\n---> Training Flat YOLOv11x Baseline (Depth 017)"
yolo train \
    data=$WORKSPACE/tier_yolo_flat_specialists/017/train.yaml \
    model=$BASE_MODEL \
    project=$MODEL_DIR/ultralytics_yolov11x_flat_level_leaf \
    name=$VERSION \
    epochs=$FLAT_EPOCHS \
    workers=$WORKERS \
    batch=$BATCH

# ==========================================
# 3. YOLOv11x Hierarchical
# ==========================================
echo -e "\n---> Training Hierarchical YOLOv11x"
python -m hierarchical_yolo.train \
    --workspace_dir $WORKSPACE \
    --model_dir $MODEL_DIR \
    --project_name ultralytics_yolov11x_hierarchical/$VERSION \
    --base_model $BASE_MODEL \
    --shallow_epochs $SHALLOW_EPOCHS \
    --final_epochs $FINAL_EPOCHS \
    --workers $WORKERS \
    --batch $BATCH

echo -e "\n✅ All YOLOv11x training tasks dispatched!"
