import os
import json
import argparse
import shutil

from pycocowriter.cocomerge import coco_filter_categories

def main():
    """
    Executes the species-only filtering pipeline for the GFISHER dataset.
    
    Phases:
    1. Loads the master hierarchy from a completed GFISHER staging directory.
    2. Identifies all species-level categories (exactly two words).
    3. Filters train/val/test JSONs to drop non-species annotations and their orphaned images.
    4. Reconstructs a filtered staging directory with symlinked image vaults.
    """
    parser = argparse.ArgumentParser(description="GFISHER Species-Only Data Orchestrator")
    parser.add_argument(
        '--staging_source', 
        type=str, 
        required=True,
        help="Path to the output of gfisher_data_orchestrator.py (contains raw train/val/test.json)"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help="Target directory for the newly filtered staging JSONs and symlinks"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("🔬 Initiating Species-Only Dataset Filter (Staging)")
    print(f"Staging Source: {args.staging_source}")
    print(f"Filtered Staging Output: {args.output_dir}")
    print("=" * 60)

    # Phase 1: Load hierarchy and extract species names (Exactly two words)
    hierarchy_path = os.path.join(args.staging_source, 'hierarchy.json')
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Missing master hierarchy at {hierarchy_path}")

    with open(hierarchy_path, 'r') as f:
        hierarchy = json.load(f)
    
    all_categories = set(hierarchy.keys()).union(set(hierarchy.values()))
    species_names = sorted([cat for cat in all_categories if len(cat.split()) == 2])
    
    print(f"\n--- Phase 1: Taxonomy Filtering ---")
    print(f"Scanned {len(all_categories)} total taxa.")
    print(f"Identified {len(species_names)} species-level (binomial) categories.")

    # Phase 2: Filter the JSON splits
    print(f"\n--- Phase 2: Purging Non-Species Annotations & Orphans ---")
    splits = ['train.json', 'val.json', 'test.json']
    for split in splits:
        in_path = os.path.join(args.staging_source, split)
        out_path = os.path.join(args.output_dir, split)
        
        if not os.path.exists(in_path):
            print(f"Warning: {split} not found. Skipping...")
            continue
            
        with open(in_path, 'r') as f:
            coco_dict = json.load(f)
            
        orig_img_count = len(coco_dict.get('images', []))
        orig_ann_count = len(coco_dict.get('annotations', []))
        
        filtered_dict = coco_filter_categories(coco_dict, species_names)
        
        new_img_count = len(filtered_dict.get('images', []))
        new_ann_count = len(filtered_dict.get('annotations', []))
        
        print(f"Processed {split}:")
        print(f"  -> Annotations: {orig_ann_count} -> {new_ann_count} (Dropped {orig_ann_count - new_ann_count})")
        print(f"  -> Images:      {orig_img_count} -> {new_img_count} (Dropped {orig_img_count - new_img_count} true orphans)")
        
        with open(out_path, 'w') as f:
            json.dump(filtered_dict, f)

    # Phase 3: Copy hierarchy and safely symlink image directories
    print(f"\n--- Phase 3: Materializing Filtered Staging Directory ---")
    shutil.copy(hierarchy_path, os.path.join(args.output_dir, 'hierarchy.json'))
    print("Master hierarchy copied.")
    
    # We must symlink the image folders so the downstream orchestrator can resolve the physical files
    for split_dir in ['train', 'val', 'test']:
        src_img_dir = os.path.abspath(os.path.join(args.staging_source, split_dir))
        dst_img_dir = os.path.join(args.output_dir, split_dir)
        
        if os.path.exists(src_img_dir):
            if os.path.islink(dst_img_dir):
                os.unlink(dst_img_dir)
            if not os.path.exists(dst_img_dir):
                os.symlink(src_img_dir, dst_img_dir)
                print(f"Symlinked physical image vault: {split_dir}")

    print("\n" + "=" * 60)
    print(f"✅ Species-Only Staging Complete! The staging directory is ready at: {args.output_dir}")
    print("Next step: Pass this directory to the generic hierarchical_yolo/data_orchestrator.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
