import os
import json
import argparse
import urllib.request
import zipfile
import io

import pycocowriter.coco2yolo

CFD_JSON_ZIP_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/community_fish_detection_dataset.json.zip"

def main():
    parser = argparse.ArgumentParser(description="Community Fish Detection Data Orchestrator")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.path.expanduser('~/datasets/cfd'),
        help="Target directory for the processed dataset (default: ~/datasets/cfd)"
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("🐟 Initiating Community Fish Detection Staging Pipeline")
    print(f"Target Directory: {data_dir}")
    print("=" * 60)

    # Phase 1: Fetch and Extract in memory
    print("\n--- Phase 1: Fetching and Extracting Annotations ---")
    print(f"Downloading CFD Zip: {CFD_JSON_ZIP_URL}")
    req = urllib.request.urlopen(CFD_JSON_ZIP_URL)
    
    with zipfile.ZipFile(io.BytesIO(req.read())) as z:
        # Assuming the JSON file inside has the exact same name as the zip minus .zip
        json_filename = 'community_fish_detection_dataset.json'
        print(f"Extracting {json_filename} from archive...")
        with z.open(json_filename) as f:
            cfd_coco = json.load(f)

    # Phase 2: Category Remapping
    print("\n--- Phase 2: Category Remapping ---")
    print("Mapping 'fish' -> 'Chordata' and dropping 'empty'...")
    
    # Update category name
    for cat in cfd_coco.get('categories', []):
        if cat['name'].lower() == 'fish':
            cat['name'] = 'Chordata'
            
    # Note: We keep category 'empty' (id 0) for now, or filter it depending on YOLO needs.
    # Usually, YOLO handles background automatically, so native empty images are fine.

    # Phase 3: Train/Val Consolidation
    print("\n--- Phase 3: Dataset Splitting & URL Injection ---")
    
    # Inject coco_url so pycocowriter can download the images natively
    print("Injecting Azure blob URLs into image metadata...")
    base_azure_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/"
    for img in cfd_coco['images']:
        img['coco_url'] = base_azure_url + img['file_name']

    # For now, we are dumping EVERYTHING into a single train.json (see discussion).
    train_path = os.path.join(data_dir, "train.json")
    
    print(f"Saving all {len(cfd_coco['images'])} images to Train Split: {train_path}")
    with open(train_path, 'w') as f:
        json.dump(cfd_coco, f)

    # Phase 4: Image Materialization
    print("\n--- Phase 4: Image Materialization ---")
    print("Executing concurrent image downloads via pycocowriter...")
    pycocowriter.coco2yolo.download_coco_images(data_dir, data_dir)

    print("\n" + "=" * 60)
    print(f"✅ CFD JSON Pre-Processing Complete! The staging directory is ready at: {data_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
