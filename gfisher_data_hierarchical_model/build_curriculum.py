import os
import json
import shutil

import pycocowriter.coco2yolo
from hierarchical_loss import hierarchy_coco_utils

# ==========================================
# Hard-coded Pipeline Constants
# ==========================================
DATASETS = os.path.expanduser('~/datasets')
DATA = os.path.join(DATASETS, 'gfisher')
GFISHER_MODELS = 'gfisher_data_hierarchical_model/models'
HIERARCHY_JSON = os.path.join(GFISHER_MODELS, 'hierarchy.json')

# New distinct directory to prevent collision with hierarchical curricula
FLAT_MODELS_DIR = os.path.join(DATA, 'alternate_depth_flat_models')


def enforce_symlinks(json_paths: list[str], src_data_dir: str, depth_dest_dir: str) -> None:
    """Safely constructs image directory symlinks pointing back to the master data pool."""
    print("  -> Enforcing Symlinks for image directories...")
    for json_path in json_paths:
        basename = os.path.splitext(os.path.basename(json_path))[0]
        
        master_img_dir = os.path.abspath(os.path.join(src_data_dir, basename, "images"))
        yolo_img_dir = os.path.join(depth_dest_dir, basename, "images")
        
        os.makedirs(master_img_dir, exist_ok=True)
        os.makedirs(os.path.dirname(yolo_img_dir), exist_ok=True)
        
        if os.path.isdir(yolo_img_dir) and not os.path.islink(yolo_img_dir):
            shutil.rmtree(yolo_img_dir)
            
        if not os.path.exists(yolo_img_dir):
            os.symlink(master_img_dir, yolo_img_dir)


def generate_flat_baseline(
    current_depth: int, 
    all_json_paths: list[str], 
    lineages: dict, 
    name_to_id: dict, 
    master_categories: list
) -> None:
    """Builds a fully self-contained, densely indexed YOLO dataset for a specific phylogenetic depth."""
    print(f"\n{'='*50}\nBuilding Flat Baseline: Depth {current_depth}\n{'='*50}")
    
    depth_dest_dir = os.path.join(FLAT_MODELS_DIR, f"{current_depth:03d}")
    staging_dir = os.path.join(depth_dest_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)

    # 1. Map and Cast all splits to the current depth
    depth_map = hierarchy_coco_utils.build_depth_map(lineages, current_depth, name_to_id)
    
    casted_cocos = {}
    for path in all_json_paths:
        with open(path, 'r') as f:
            coco_dict = json.load(f)
        casted_cocos[path] = hierarchy_coco_utils.cast_coco_to_depth(coco_dict, depth_map)

    # 2. Gather active IDs globally so Train/Val/Test share the exact same ID mapping
    active_ids = hierarchy_coco_utils.get_active_category_ids(*casted_cocos.values())
    old_to_new, new_to_old = hierarchy_coco_utils.build_dense_category_map(active_ids)

    # 3. Restrict, Re-index, and Stage the JSONs
    for path, casted_coco in casted_cocos.items():
        final_coco = hierarchy_coco_utils.restrict_and_reindex_coco(casted_coco, old_to_new, master_categories)
        with open(os.path.join(staging_dir, os.path.basename(path)), 'w') as f:
            json.dump(final_coco, f)

    # 4. Serialize the Global Mapping Artifacts
    map_path = os.path.join(depth_dest_dir, "category_index_map.json")
    with open(map_path, 'w') as f:
        # We save both mappings. YOLO classes are 0-indexed internally, 
        # so YOLO class ID is exactly (Flat COCO ID - 1)
        mapping_output = {
            "flat_coco_id_to_master_coco_id": new_to_old,
            "yolo_class_id_to_master_coco_id": {k - 1: v for k, v in new_to_old.items()}
        }
        json.dump(mapping_output, f, indent=4)

    # 5. Convert to YOLO Format
    enforce_symlinks(all_json_paths, DATA, depth_dest_dir)
    print("  -> Running Pycocowriter Conversion...")
    pycocowriter.coco2yolo.coco2yolo(staging_dir, depth_dest_dir)
    
    shutil.rmtree(staging_dir)
    print(f"Depth {current_depth} completed successfully.")


def main():
    print("Initializing Flat Baseline Generation...")
    
    with open(HIERARCHY_JSON, 'r') as f:
        hierarchy_tree = json.load(f)

    lineages = hierarchy_coco_utils.build_all_lineages(hierarchy_tree)
    max_depth = max(len(lin) for lin in lineages.values()) - 1
    
    split_files = pycocowriter.coco2yolo.discover_coco_files(DATA)
    all_json_paths = split_files['train'] + split_files['val'] + split_files['test']
    
    if not all_json_paths:
        print(f"Error: No COCO JSON files found in {DATA}.")
        return

    with open(all_json_paths[0], 'r') as f:
        reference_coco = json.load(f)
        
    master_categories = reference_coco.get('categories', [])
    name_to_id = {cat['name']: cat['id'] for cat in master_categories}

    # Generate baselines iteratively
    for current_depth in range(max_depth + 1):
        generate_flat_baseline(
            current_depth, 
            all_json_paths, 
            lineages, 
            name_to_id, 
            master_categories
        )

if __name__ == "__main__":
    main()
