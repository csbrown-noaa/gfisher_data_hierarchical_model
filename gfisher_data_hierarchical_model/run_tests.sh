#!/bin/bash

# ==============================================================================
# HIERARCHICAL YOLO EVALUATION MATRIX ROUTER (WET VERSION)
# 6-Axis Organization:
# [Scale] -> [Architecture] -> [Train Dataset] -> [Train Depth] -> [Test Dataset] -> [Test Depth]
# ==============================================================================

EVAL_ROOT="./evaluation_matrix"
mkdir -p "$EVAL_ROOT"

# ==============================================================================
# 8N SCALE MODELS
# ==============================================================================

# ------------------------------------------------------------------------------
# RUN 1: Phase 1 Objectness Test (Flat Objectness Specialist)
# Axes: [8n] x [flat] x [train_full_taxonomy] x [train_depth_000] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 1: 8n | flat | train_full_taxonomy | train_depth_000 | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_full_taxonomy/train_depth_000/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
    --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_000/1.0.0/weights/best.pt" \
    --model_type "flat" \
    --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/000/train.yaml" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 2: Phase 1 Objectness Test (Flat Bottom-Up Aggregator)
# Axes: [8n] x [flat] x [train_full_taxonomy] x [train_depth_017] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 2: 8n | flat | train_full_taxonomy | train_depth_017 | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_full_taxonomy/train_depth_017/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
    --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
    --model_type "flat" \
    --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 3: Phase 1 Objectness Test (Hierarchical Omnivorous)
# Axes: [8n] x [hierarchical] x [train_full_taxonomy] x [train_depth_all] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 3: 8n | hierarchical | train_full_taxonomy | train_depth_all | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_full_taxonomy/train_depth_all/test_full_taxonomy/eval_depth_000"
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
# RUN 4: Phase 3 Toxicity Test (Model A: Flat Full Evaluated on Species)
# Axes: [8n] x [flat] x [train_full_taxonomy] x [train_depth_017] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 4: 8n | flat | train_full_taxonomy | train_depth_017 | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_full_taxonomy/train_depth_017/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
    model="$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
    data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    split="test" \
    project="$OUT_DIR" \
    name="metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 5: Phase 3 Toxicity/Specificity Test (Model B: Flat Species Evaluated on Species)
# Axes: [8n] x [flat] x [train_species_only] x [train_depth_017] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 5: 8n | flat | train_species_only | train_depth_017 | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_species_only/train_depth_017/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
    model="$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
    data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    split="test" \
    project="$OUT_DIR" \
    name="metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 6: Phase 3 Specificity/Benefit Test (Model A: Hier Species Evaluated on Species)
# Axes: [8n] x [hierarchical] x [train_species_only] x [train_depth_all] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 6: 8n | hierarchical | train_species_only | train_depth_all | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_species_only/train_depth_all/test_species_only/eval_depth_017"
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
# RUN 7: Phase 3 Hierarchy Benefit Test (Model B: Hier Full Evaluated on Species)
# Axes: [8n] x [hierarchical] x [train_full_taxonomy] x [train_depth_all] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 7: 8n | hierarchical | train_full_taxonomy | train_depth_all | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_full_taxonomy/train_depth_all/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
    --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt" \
    --hierarchical_eval_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_full_head/017/train.yaml" \
    --flat_data_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    --hierarchy_json "$HOME/datasets/gfisher_species_only_hierarchical_workspace/hierarchy.json" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 8: The "Distractor" Test (Model A: Flat Species Evaluated on Full Taxonomy)
# Axes: [8n] x [flat] x [train_species_only] x [train_depth_017] x [test_full_taxonomy] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 8: 8n | flat | train_species_only | train_depth_017 | test_full_taxonomy | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/flat/train_species_only/train_depth_017/test_full_taxonomy/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
    model="$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_flat_level_leaf/1.0.0/weights/best.pt" \
    data="$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    split="test" \
    project="$OUT_DIR" \
    name="metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 9: The "Distractor" Test (Model B: Hier Species Evaluated on Full Taxonomy)
# Axes: [8n] x [hierarchical] x [train_species_only] x [train_depth_all] x [test_full_taxonomy] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 9: 8n | hierarchical | train_species_only | train_depth_all | test_full_taxonomy | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov8n/hierarchical/train_species_only/train_depth_all/test_full_taxonomy/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
    --weights "$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov8n_hierarchical/1.0.0/weights/best.pt" \
    --hierarchical_eval_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml" \
    --flat_data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    --hierarchy_json "$HOME/datasets/gfisher_workspace/hierarchy.json" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"


# ==============================================================================
# 11X SCALE MODELS
# ==============================================================================

# ------------------------------------------------------------------------------
# RUN 10: Phase 1 Objectness Test (Flat Objectness Specialist)
# Axes: [11x] x [flat] x [train_full_taxonomy] x [train_depth_000] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 10: 11x | flat | train_full_taxonomy | train_depth_000 | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_full_taxonomy/train_depth_000/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
    --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_000/1.0.0/weights/best.pt" \
    --model_type "flat" \
    --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/000/train.yaml" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 11: Phase 1 Objectness Test (Flat Bottom-Up Aggregator)
# Axes: [11x] x [flat] x [train_full_taxonomy] x [train_depth_017] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 11: 11x | flat | train_full_taxonomy | train_depth_017 | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_full_taxonomy/train_depth_017/test_full_taxonomy/eval_depth_000"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks objectness \
    --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
    --model_type "flat" \
    --data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 12: Phase 1 Objectness Test (Hierarchical Omnivorous)
# Axes: [11x] x [hierarchical] x [train_full_taxonomy] x [train_depth_all] x [test_full_taxonomy] x [eval_depth_000]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 12: 11x | hierarchical | train_full_taxonomy | train_depth_all | test_full_taxonomy | eval_depth_000"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_full_taxonomy/train_depth_all/test_full_taxonomy/eval_depth_000"
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
# RUN 13: Phase 3 Toxicity Test (Model A: Flat Full Evaluated on Species)
# Axes: [11x] x [flat] x [train_full_taxonomy] x [train_depth_017] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 13: 11x | flat | train_full_taxonomy | train_depth_017 | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_full_taxonomy/train_depth_017/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
    model="$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
    data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    split="test" \
    project="$OUT_DIR" \
    name="metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 14: Phase 3 Toxicity/Specificity Test (Model B: Flat Species Evaluated on Species)
# Axes: [11x] x [flat] x [train_species_only] x [train_depth_017] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 14: 11x | flat | train_species_only | train_depth_017 | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_species_only/train_depth_017/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
    model="$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
    data="$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    split="test" \
    project="$OUT_DIR" \
    name="metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 15: Phase 3 Specificity/Benefit Test (Model A: Hier Species Evaluated on Species)
# Axes: [11x] x [hierarchical] x [train_species_only] x [train_depth_all] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 15: 11x | hierarchical | train_species_only | train_depth_all | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_species_only/train_depth_all/test_species_only/eval_depth_017"
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
# RUN 16: Phase 3 Hierarchy Benefit Test (Model B: Hier Full Evaluated on Species)
# Axes: [11x] x [hierarchical] x [train_full_taxonomy] x [train_depth_all] x [test_species_only] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 16: 11x | hierarchical | train_full_taxonomy | train_depth_all | test_species_only | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_full_taxonomy/train_depth_all/test_species_only/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
    --weights "$HOME/Models/hierarchical_yolo_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt" \
    --hierarchical_eval_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_full_head/017/train.yaml" \
    --flat_data_yaml "$HOME/datasets/gfisher_species_only_hierarchical_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    --hierarchy_json "$HOME/datasets/gfisher_species_only_hierarchical_workspace/hierarchy.json" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 17: The "Distractor" Test (Model A: Flat Species Evaluated on Full Taxonomy)
# Axes: [11x] x [flat] x [train_species_only] x [train_depth_017] x [test_full_taxonomy] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 17: 11x | flat | train_species_only | train_depth_017 | test_full_taxonomy | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/flat/train_species_only/train_depth_017/test_full_taxonomy/eval_depth_017"
mkdir -p "$OUT_DIR"

yolo val \
    model="$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov11x_flat_level_leaf/1.0.0/weights/best.pt" \
    data="$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    split="test" \
    project="$OUT_DIR" \
    name="metrics" > "$OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# RUN 18: The "Distractor" Test (Model B: Hier Species Evaluated on Full Taxonomy)
# Axes: [11x] x [hierarchical] x [train_species_only] x [train_depth_all] x [test_full_taxonomy] x [eval_depth_017]
# ------------------------------------------------------------------------------
echo -e "\n▶️ EXECUTING RUN 18: 11x | hierarchical | train_species_only | train_depth_all | test_full_taxonomy | eval_depth_017"
OUT_DIR="$EVAL_ROOT/ultralytics_yolov11x/hierarchical/train_species_only/train_depth_all/test_full_taxonomy/eval_depth_017"
mkdir -p "$OUT_DIR"

python -m hierarchical_yolo.apples2apples_benchmarks specificity \
    --weights "$HOME/Models/hierarchical_yolo_species_only_models/ultralytics_yolov11x_hierarchical/1.0.0/weights/best.pt" \
    --hierarchical_eval_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_full_head/017/train.yaml" \
    --flat_data_yaml "$HOME/datasets/gfisher_workspace/tier_yolo_flat_specialists/017/train.yaml" \
    --hierarchy_json "$HOME/datasets/gfisher_workspace/hierarchy.json" \
    --split "test" \
    --project "$OUT_DIR" \
    --name "metrics" > "$OUT_DIR/run.log"

echo -e "\n🎉 All 18 explicit matrix combinations dispatched and completed!"
