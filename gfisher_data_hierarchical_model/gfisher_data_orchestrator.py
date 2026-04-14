import os
import json
import glob
import argparse

# Import the orchestrator functions based on our pipeline architecture
try:
    from coco_worms_expansion import expand_and_align_dataset
except ImportError:
    from hierarchical_loss.worms_expander import expand_and_align_dataset

from hierarchical_yolo.hierarchical_curriculum_builder import build_hierarchical_curriculum
from hierarchical_yolo.flat_baseline_builder import build_flat_baselines
import pycocowriter.coco2yolo
from pycocowriter.coco_split_utils import rarity_stratified_split

# ==========================================
# GFISHER Dataset Configurations
# ==========================================
TRAIN_DATA_URL = "https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/train_annotations_worms.json"
TEST_DATA_URL = "https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/test_annotations_worms.json"

def main():
    """
    Executes the end-to-end data preparation pipeline for the GFISHER dataset.
    
    Phases:
    1. Fetches raw COCO data from GCP and aligns it to the WoRMS taxonomy.
    2. Performs Rarity-Stratified Split to generate Validation set.
    3. Performs Anchor YOLO Conversion to materialize/download image files.
    """
    parser = argparse.ArgumentParser(description="GFISHER End-to-End Data Orchestrator")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.path.expanduser('~/datasets/gfisher'),
        help="Target directory for the processed dataset (default: ~/datasets/gfisher)"
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Initiating GFISHER Data Staging Pipeline")
    print(f"Target Directory: {data_dir}")
    print("=" * 60)

    # Step 1: Fetch, expand taxonomy, and align the datasets
    # This automatically categorizes them into 'train' and 'test' based on the URL strings
    print("\n--- Phase 1: Ingestion & WoRMS Alignment ---")
    expand_and_align_dataset(
        data_dir=data_dir, 
        coco_sources=[TRAIN_DATA_URL, TEST_DATA_URL]
    )

    # Step 2: The Strategic Chokepoint (Train/Val Split)
    print("\n--- Phase 2: Rarity Stratified Split (Train/Val) ---")
    # 1. Locate the freshly generated training JSON
    train_json_pattern = os.path.join(data_dir, "train_*_aligned.json")
    train_files = glob.glob(train_json_pattern)
    
    if not train_files:
        raise FileNotFoundError(f"Could not find aligned training JSON matching {train_json_pattern}. Did Phase 1 fail?")
        
    original_train_json_path = train_files[0]
    print(f"Intercepting aligned data: {original_train_json_path}")
    
    # 2. Load the JSON into memory
    with open(original_train_json_path, 'r') as f:
        train_coco_dict = json.load(f)
        
    # 3. Execute the Sequence-Aware Rarity Split
    print("Executing sequential, rarity-stratified 85/15 split...")
    train_split, val_split = rarity_stratified_split(
        coco_dict=train_coco_dict, 
        split_ratios=[0.85, 0.15], 
        sort_by_filename=True,   # CRITICAL: Mitigates video data leakage!
        seed=42                  # Locks the split mathematically across all runs
    )
    
    # 4. Save the physically separated dictionaries
    new_train_path = os.path.join(data_dir, "train_stratified.json")
    new_val_path = os.path.join(data_dir, "val_stratified.json")
    
    print(f"Saving Train Split ({len(train_split['images'])} images) to {new_train_path}")
    with open(new_train_path, 'w') as f:
        json.dump(train_split, f)
        
    print(f"Saving Val Split ({len(val_split['images'])} images) to {new_val_path}")
    with open(new_val_path, 'w') as f:
        json.dump(val_split, f)
        
    # 5. ASSASSINATE THE GHOST FILE
    # If we don't do this, Pycocowriter will find it and merge it with our new splits,
    # doubling the dataset and leaking validation images back into training.
    print(f"Deleting original ghost file to prevent double-dipping: {original_train_json_path}")
    os.remove(original_train_json_path)

    # Step 3: Anchor YOLO Conversion (Forces Image Downloads to Local Disk)
    print("\n--- Phase 3: Anchor YOLO Conversion (Materializing Images) ---")
    # Filter out empty splits to prevent pycocowriter loops
    pycocowriter.coco2yolo.coco2yolo(data_dir, data_dir)

    print("\n" + "=" * 60)
    print(f"✅ GFISHER Pre-Processing Complete! The staging directory is ready at: {data_dir}")
    print("Next step: Pass this directory to the generic hierarchical_yolo/data_orchestrator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
