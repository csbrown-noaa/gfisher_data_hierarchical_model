#!/bin/bash

# ==============================================================================
# HIERARCHICAL YOLO EVALUATION MATRIX (UNROLLED)
#
# Matrix Axes:
# 1. Scale: 8n | 11x
# 2. Architecture: flat | hierarchical
# 3. Training Domain: train_full_taxonomy | train_species_only
# 4. Evaluation Domain: test_full_taxonomy | test_species_only
# 5. Taxonomic Depth: eval_depth_000 (Root/Objectness) | eval_depth_017 (Leaf)
# ==============================================================================

EVAL_ROOT="./evaluation_matrix"
echo "🚀 Starting Explicit Evaluation Matrix..."


# ==============================================================================
# 8N SCALE MODELS
# ==============================================================================

# ------------------------------------------------------------------------------
# RUN 1: Phase 1 Objectness (Test A)
# Axes: [8n] x [flat] x [train_full_taxonomy] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 1: 8n | flat | train_full_taxonomy | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_full_taxonomy/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_000/1.0.0/weights/best.pt" \
     --model_type "flat" \
     --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/000/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 2: Phase 1 Objectness (Test A)
# Axes: [8n] x [hierarchical] x [train_full_taxonomy] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 2: 8n | hierarchical | train_full_taxonomy | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_full_taxonomy/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt" \
     --model_type "hierarchical" \
     --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 3: Phase 3 Toxicity Test (Model A: Flat Full Evaluated on Species)
# Axes: [8n] x [flat] x [train_full_taxonomy] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 3: 8n | flat | train_full_taxonomy | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_full_taxonomy/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
     model="$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
     data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     split="test" \
     project="$OUT_DIR" \
     name="metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 4: Phase 3 Toxicity Test (Model B) & Clean Specificity (Model B)
# Axes: [8n] x [flat] x [train_species_only] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 4: 8n | flat | train_species_only | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_species_only/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
     model="$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
     data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     split="test" \
     project="$OUT_DIR" \
     name="metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 5: Phase 3 Clean Specificity (Model A) & Hierarchy Benefit Test (Model B)
# Axes: [8n] x [hierarchical] x [train_species_only] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 5: 8n | hierarchical | train_species_only | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_species_only/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
     --weights "$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt" \
     --hierarchical_eval_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_full_head/017/train.yaml" \
     --flat_data_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_species_only_hierarchical_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 6: Phase 3 Hierarchy Benefit Test (Model A: Hier Full Evaluated on Species)
# Axes: [8n] x [hierarchical] x [train_full_taxonomy] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 6: 8n | hierarchical | train_full_taxonomy | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_full_taxonomy/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt" \
     --hierarchical_eval_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_full_head/017/train.yaml" \
     --flat_data_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_species_only_hierarchical_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ==============================================================================
# 11X SCALE MODELS
# ==============================================================================

# ------------------------------------------------------------------------------
# RUN 7: Phase 1 Objectness (Test A)
# Axes: [11x] x [flat] x [train_full_taxonomy] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 7: 11x | flat | train_full_taxonomy | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_full_taxonomy/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_000/1.0.0/weights/best.pt" \
     --model_type "flat" \
     --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/000/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 8: Phase 1 Objectness (Test A)
# Axes: [11x] x [hierarchical] x [train_full_taxonomy] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 8: 11x | hierarchical | train_full_taxonomy | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_full_taxonomy/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt" \
     --model_type "hierarchical" \
     --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 9: Phase 3 Toxicity Test (Model A: Flat Full Evaluated on Species)
# Axes: [11x] x [flat] x [train_full_taxonomy] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 9: 11x | flat | train_full_taxonomy | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_full_taxonomy/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
     model="$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
     data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     split="test" \
     project="$OUT_DIR" \
     name="metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 10: Phase 3 Toxicity Test (Model B) & Clean Specificity (Model B)
# Axes: [11x] x [flat] x [train_species_only] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 10: 11x | flat | train_species_only | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_species_only/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
     model="$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
     data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     split="test" \
     project="$OUT_DIR" \
     name="metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 11: Phase 3 Clean Specificity (Model A) & Hierarchy Benefit Test (Model B)
# Axes: [11x] x [hierarchical] x [train_species_only] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 11: 11x | hierarchical | train_species_only | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_species_only/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
     --weights "$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt" \
     --hierarchical_eval_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_full_head/017/train.yaml" \
     --flat_data_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_species_only_hierarchical_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"


# ------------------------------------------------------------------------------
# RUN 12: Phase 3 Hierarchy Benefit Test (Model A: Hier Full Evaluated on Species)
# Axes: [11x] x [hierarchical] x [train_full_taxonomy] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 12: 11x | hierarchical | train_full_taxonomy | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_full_taxonomy/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
     --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt" \
     --hierarchical_eval_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_full_head/017/train.yaml" \
     --flat_data_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
     --hierarchy_json "$HOME/datasets/gfisher_species_only_hierarchical_workspace/hierarchy.json" \
     --split "test" \
     --project "$OUT_DIR" \
     --name "metrics" > "$OUT_DIR/run.log"

echo -e "\n🎉 All required explicit matrix combinations dispatched and completed!"

