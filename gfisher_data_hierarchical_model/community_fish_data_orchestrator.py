import os
import json
import argparse
import urllib.request
import zipfile
import io

CFD_JSON_ZIP_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/community_fish_detection_dataset.json.zip"

def main():
    parser = argparse.ArgumentParser(description="Community Fish Detection Data Orchestrator")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.path.expanduser('~/datasets/community_fish_staging'),
        help="Target directory for the processed dataset (default: ~/datasets/community_fish_staging)"
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
    
    # Validate strict dataset assumptions
    categories = cfd_coco.get('categories', [])
    assert len(categories) == 2, f"Expected exactly 2 categories ('fish', 'empty'), found {len(categories)}"
    
    cat_names = {c['name']: c['id'] for c in categories}
    assert 'fish' in cat_names, "Missing expected category: 'fish'"
    assert 'empty' in cat_names, "Missing expected category: 'empty'"
    
    empty_id = cat_names['empty']
    fish_id = cat_names['fish']

    # Update categories: keep only 'fish' and remap its name to 'Chordata'
    cfd_coco['categories'] = [{'id': fish_id, 'name': 'Chordata'}]
    
    # Filter out dummy 'empty' annotations
    print(f"Purging dummy annotations for 'empty' category (ID: {empty_id})...")
    original_ann_count = len(cfd_coco.get('annotations', []))
    cfd_coco['annotations'] = [
        ann for ann in cfd_coco.get('annotations', []) 
        if ann.get('category_id') != empty_id
    ]
    
    removed_count = original_ann_count - len(cfd_coco['annotations'])
    print(f"Successfully dropped {removed_count} dummy annotations.")

    # Phase 3 & 4: Dataset Splitting, URL Injection, and Image Materialization
    print("\n--- Phase 3 & 4: Metadata Flattening & Image Materialization ---")
    
    try:
        from google.cloud.storage import Client, transfer_manager
    except ImportError:
        raise ImportError("❌ The 'google-cloud-storage' package is required. Run: pip install google-cloud-storage")

    print("Connecting to public GCP bucket anonymously...")
    # Anonymous client completely bypasses local credential requests!
    storage_client = Client.create_anonymous_client()
    bucket = storage_client.bucket("public-datasets-lila")

    print("Flattening filenames in JSON and preparing download queue...")
    blob_file_pairs = []
    
    # Base azure url for coco_url injection (just in case you need standard COCO URLs downstream)
    base_azure_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/"
    
    for img in cfd_coco['images']:
        original_file_name = img['file_name']  # e.g., 'JPEGImages/torsi_20190716-021037.129.JPG'
        
        # Inject coco_url to maintain standard COCO conventions
        img['coco_url'] = base_azure_url + original_file_name
        
        # Flatten the filename for YOLO compatibility (strip 'JPEGImages/' entirely)
        flat_name = original_file_name.replace('JPEGImages/', '').replace('/', '_').replace('\\', '_')
        img['file_name'] = flat_name
        
        # Setup the direct download mapping (GCP Source Blob -> Flat Local File)
        blob_path = f"community-fish-detection-dataset/{original_file_name}"
        dest_path = os.path.join(data_dir, flat_name)
        
        # Only queue for download if it isn't already on disk
        if not os.path.exists(dest_path):
            blob = bucket.blob(blob_path)
            blob_file_pairs.append((blob, dest_path))

    # For now, we are dumping EVERYTHING into a single train.json (see discussion).
    train_path = os.path.join(data_dir, "train.json")
    print(f"Saving all {len(cfd_coco['images'])} images to Train Split: {train_path}")
    with open(train_path, 'w') as f:
        json.dump(cfd_coco, f)

    if blob_file_pairs:
        print(f"\nInitiating high-speed concurrent download of {len(blob_file_pairs)} images...")
        print("Using google.cloud.storage transfer_manager (handles multiplexing automatically).")
        
        # transfer_manager uses ProcessPoolExecutor by default. 
        # 32 workers is the sweet spot for maximizing bandwidth without starving CPU context switching.
        results = transfer_manager.download_many(
            blob_file_pairs,
            max_workers=32,
            raise_exception=False
        )
        
        success_count = sum(1 for result in results if not isinstance(result, Exception))
        fail_count = len(results) - success_count
        
        print(f"✅ Successfully downloaded {success_count} images.")
        if fail_count > 0:
            print(f"⚠️ Failed to download {fail_count} images. (Check your internet connection or rate limits)")
    else:
        print("\n✅ All images already exist locally. Skipping download.")

    print("\n" + "=" * 60)
    print(f"✅ CFD JSON Pre-Processing Complete! The staging directory is ready at: {data_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
