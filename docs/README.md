# GFISHER Hierarchical Model Training

Training pipelines for [hierarchical models](https://github.com/csbrown-noaa/hierarchical_yolo) trained on [the GFISHER survey](https://restoreactscienceprogram.noaa.gov/projects/reef-fish-survey) data, located [here](https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/annotations_worms.json)

# Usage

## The Architecture

This project sits on top of three core foundational libraries:

* **`pycocowriter`**: Handles graph-aware data splitting (solving the multi-label "Passenger Problem") and YOLO format conversions.

* **`hierarchical_loss`**: The math engine. Performs O(1) GPU graph traversals using dense boolean masks and computes the Probabilistic Expected Risk loss.

* **`hierarchical_yolo`**: DDP-ready overrides for the Ultralytics YOLOv8 architecture. It includes the training orchestrators that automatically stage curriculum training across the phylogenetic tree.

## Phase 1: Data Staging (Dataset-Specific)

The first phase is isolated strictly to **data preparation and downloading**. The GFISHER staging orchestrator fetches annotations, splits the raw data, harvests the taxonomic tree into memory, downloads the physical image files to a local staging directory, and finally saves the master taxonomy.

To run the staging pipeline:

```bash
python gfisher_data_orchestrator.py --data_dir ~/datasets/gfisher_staging
```

**What this does:**

1. **Fetch Raw Annotations:** Downloads the raw COCO JSONs directly from GCP into memory.
2. **Rarity-Stratified Split:** Splits the raw training data 85/15 while ensuring rare species aren't swallowed by common ones, grouping by filename to prevent video-frame leakage.
3. **WoRMS Taxonomy Harvesting:** Scans the raw category names and queries the WoRMS API to build the full taxonomic lineage in memory. *Note: The dataset categories remain in their raw, unaligned ID space at this stage.*
4. **Image Materialization:** Parses the original image URLs and directly downloads all raw images to local disk (`~/datasets/gfisher_staging/train/images`, etc.). Bypassing the full YOLO conversion wrapper avoids strict metadata assertions, allowing the staging script to operate smoothly on raw, unaligned datasets.
5. **Save Master Hierarchy:** Writes the `hierarchy.json` file to the staging directory *after* image materialization is complete. This defensive ordering prevents upstream tools from crashing when aggressively globbing for COCO files during the image download phase.

## Phase 2: Workspace Compilation (Generic)

Once the raw data is safely staged on disk, we use the generic "Compiler" from the `hierarchical_yolo` library. This script works entirely offline to build a specialized, symlink-driven workspace for hierarchical and flat ablation studies without duplicating heavy image files.

```bash
python -m hierarchical_yolo.data_orchestrator \
    --source_dir ~/datasets/gfisher_staging \
    --workspace_dir ~/datasets/gfisher_workspace
```

**What this does:**

1. **Universal Taxonomy Alignment (The Clean Room):** Uses the `HierarchicalCocoAligner` to map the raw, disjointed category IDs from the staging datasets into a mathematically contiguous, 1-to-N integer index space required by the math engine.
2. **Curriculum Generation:** Creates the `tier_yolo_full_head/` datasets mapping annotations up the tree while maintaining the full network head shape.
3. **Flat Baseline Generation:** Creates `tier_yolo_flat_specialists/` datasets. These are standard, densely-indexed YOLO datasets at specific tree depths used for the "Comparative Arena" ablation studies.

## Phase 3: Training the Hierarchical Curriculum

With the compiled workspace ready, we use the abstracted `train` orchestrator.

This module performs **Staged Curriculum Training**. It passes the weights sequentially from shallow levels of the taxonomy (e.g., broad families) down to the final deep classes (e.g., species), locking in generalized hierarchical features before attempting fine-grained classification.

To launch training, point the module to your compiled workspace:

```bash
python -m hierarchical_yolo.train \
    --data_dir ~/datasets/gfisher_workspace \
    --model_dir ~/Models/runs \
    --project_name hierarchical_gfisher_v1 \
    --base_model yolov8n.pt \
    --shallow_epochs 2 \
    --final_epochs 20
```

*Note: The trainer automatically handles DDP environment variables (`HIERARCHY_PATH`) so the custom loss function can safely distribute the tree across multiple GPUs.*

**Locating Your Final Model:** Because the curriculum trainer stages the weights through the tree, your final, fully-trained model weights will be located in the deepest run folder (zero-padded to 3 digits). For example:
`~/Models/runs/hierarchical_gfisher_v1/curriculum_depth_00X/weights/best.pt`

## Phase 4: The Comparative Arena (Flat Baselines)

To objectively evaluate the hierarchical model's performance, it is compared against standard YOLO models trained at specific taxonomic depths (e.g., a model that *only* knows about Families).

The generic orchestrator already built these datasets inside the workspace under `tier_yolo_flat_specialists/`.

To train a flat baseline (e.g., for depth 003), use the standard Ultralytics CLI. **Note:** Ensure you use the same `project` name as your curriculum run to keep your experiments grouped (see Experiment Tracking below).

```bash
yolo train data=~/datasets/gfisher_workspace/tier_yolo_flat_specialists/003/train.yaml model=yolov8n.pt project=~/Models/runs/hierarchical_gfisher_v1 name=flat_depth_003 epochs=30 imgsz=640
```

## Phase 5: Experiment Tracking & Reproducibility

To ensure scientific traceability, this pipeline utilizes an **Experiment Namespace Strategy** built on top of native Ultralytics logging.

When conducting experiments (e.g., testing new augmentations or base models), you define an experimental condition using the `--project_name` argument. The pipeline automatically manages the individual run names for each stage of the tree.

This generates a clean **Traceability Matrix** in your output directory:

```text
~/Models/runs/
└── hierarchical_gfisher_v1/       <-- The Experimental Condition (Project)
    ├── curriculum_depth_000/      <-- The Staged Runs (Name)
    ├── curriculum_depth_001/
    ├── ...
    ├── flat_depth_000/            <-- The Flat Baselines
    └── flat_depth_001/
```

**Hyperparameter Logging:** You do not need to manually record your training configurations. For every run, an immutable `args.yaml` file is automatically generated inside the run folder. This file contains the exact hyperparameters, augmentations, and learning rates used, ensuring that every weight file is scientifically reproducible. Grouping runs by `--project_name` allows tracking tools (like TensorBoard or Weights & Biases) to instantly overlay and compare loss curves across the entire taxonomy.

## Phase 6: Inference & Prediction Export

Once your hierarchical model is trained, you can run inference to generate a viewer-compatible COCO JSON.

Unlike standard NMS which blindly filters by the global argmax, this prediction module anchors at the taxonomic roots and exports the **full soft score vector (marginal probabilities)** for every surviving bounding box. This allows downstream web viewers to dynamically filter predictions at any depth of the tree.

Instead of relying on brittle manual URL prefixes, the pipeline securely routes predictions back to the original image URLs by mapping directly against the raw staging datasets.

```bash
python -m hierarchical_yolo.predict \
    --data_dir ~/datasets/gfisher_workspace \
    --source_dir ~/datasets/gfisher_staging \
    --model_dir ~/Models/runs \
    --project_name hierarchical_gfisher_v1 \
    --split val \
    --output hierarchical_val_predictions.json
```

**Arguments:**

* `--source_dir`: (Required) Path to the original data staging directory. Used to natively extract the original image URLs from the raw COCO JSONs, guaranteeing accurate web links.

* `--weights`: (Optional) Path to specific `best.pt`. If omitted, it automatically finds the latest run in the project directory.

* `--nms_iou_thres`: (Default 0.7) IoU threshold for Non-Max Suppression.

* `--nms_conf_thres`: (Default 0.01) A very permissive confidence threshold to cast a wide net for downstream soft-filtering.

The pipeline is entirely located in the included jupyter notebook.  Please reference the notebook for more information.

# Contributing

We would love to have your contributions that improve current functionality, fix bugs, or add new features.  See [the contributing guidelines](https://www.google.com/search?q=CONTRIBUTING.md) for more info.

# Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and
Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project
code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the
Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub
project will be governed by all applicable Federal law. Any reference to specific commercial products,
processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or
imply their endorsement, recommendation or favoring by the Department of Commerce. The Department
of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to
imply endorsement of any commercial product or activity by DOC or the United States Government.

