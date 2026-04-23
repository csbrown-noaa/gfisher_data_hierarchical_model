import os
import json
import argparse
import urllib.request

from hierarchical_loss.worms_expander import WormsTaxonomyProvider
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
    1. Fetches raw COCO data from GCP.
    2. Performs Rarity-Stratified Split to generate Validation set.
    3. Harvests the WoRMS taxonomy into memory.
    4. Materializes/downloads image files from COCO URLs.
    5. Saves the master hierarchy tree to avoid pycocowriter parsing conflicts.
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

    # Step 1: Fetch Raw Data
    print("\n--- Phase 1: Fetching Raw Annotations ---")
    print(f"Downloading Training Data: {TRAIN_DATA_URL}")
    train_req = urllib.request.urlopen(TRAIN_DATA_URL)
    train_coco_dict = json.loads(train_req.read())

    print(f"Downloading Testing Data: {TEST_DATA_URL}")
    test_req = urllib.request.urlopen(TEST_DATA_URL)
    test_coco_dict = json.loads(test_req.read())

    # Step 2: The Strategic Chokepoint (Train/Val Split)
    print("\n--- Phase 2: Rarity Stratified Split (Train/Val) ---")
    print("Executing sequential, rarity-stratified 85/15 split on raw training data...")
    train_split, val_split = rarity_stratified_split(
        coco_dict=train_coco_dict, 
        split_ratios=[0.85, 0.15], 
        sort_by_filename=True,   # CRITICAL: Mitigates video data leakage!
        seed=42                  # Locks the split mathematically across all runs
    )
    
    # Save the physically separated dictionaries
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")
    test_path = os.path.join(data_dir, "test.json")
    
    print(f"Saving Train Split ({len(train_split['images'])} images) to {train_path}")
    with open(train_path, 'w') as f:
        json.dump(train_split, f)
        
    print(f"Saving Val Split ({len(val_split['images'])} images) to {val_path}")
    with open(val_path, 'w') as f:
        json.dump(val_split, f)
        
    print(f"Saving Test Split ({len(test_coco_dict['images'])} images) to {test_path}")
    with open(test_path, 'w') as f:
        json.dump(test_coco_dict, f)

    # Step 3: Taxonomy Harvesting
    print("\n--- Phase 3: WoRMS Taxonomy Harvesting ---")
    print("Scanning categories and querying WoRMS API for full lineage...")
    provider = WormsTaxonomyProvider(use_cache=True)
    # The provider handles combining unique classes internally
    provider.build_master_hierarchy(train_split, val_split, test_coco_dict)

    # Step 4: Image Materialization (Direct Download)
    print("\n--- Phase 4: Image Materialization ---")
    # This directly parses train/val/test JSONs to download their images locally.
    # By bypassing the full coco2yolo conversion wrapper, we avoid triggering strict YOLO
    # metadata assertions on our raw, unaligned datasets.
    pycocowriter.coco2yolo.download_coco_images(data_dir, data_dir)

    # Step 5: Save Hierarchy
    print("\n--- Phase 5: Saving Master Hierarchy ---")
    hierarchy_path = os.path.join(data_dir, "hierarchy.json")
    with open(hierarchy_path, 'w') as f:
        json.dump(provider.hierarchy_tree, f, indent=4)
    print(f"Master hierarchy tree saved to: {hierarchy_path}")

    print("\n" + "=" * 60)
    print(f"✅ GFISHER Pre-Processing Complete! The staging directory is ready at: {data_dir}")
    print("Next step: Pass this directory to the generic hierarchical_yolo/data_orchestrator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
