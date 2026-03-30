import os

# Import the orchestrator functions based on our pipeline architecture
try:
    from coco_worms_expansion import expand_and_align_dataset
except ImportError:
    from hierarchical_loss.worms_expander import expand_and_align_dataset

from hierarchical_yolo.hierarchical_curriculum_builder import build_hierarchical_curriculum
from hierarchical_yolo.flat_baseline_builder import build_flat_baselines
import pycocowriter.coco2yolo

# ==========================================
# GFISHER Dataset Configurations
# ==========================================
DATA_DIR = os.path.expanduser('~/datasets/gfisher')

TRAIN_DATA_URL = "https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/train_annotations_worms.json"
TEST_DATA_URL = "https://storage.googleapis.com/nmfs_odp_hq/nodd_tools/datasets/gfisher/test_annotations_worms.json"

def main():
    """
    Executes the end-to-end data preparation pipeline for the GFISHER dataset.
    
    Phases:
    1. Fetches raw COCO data from GCP and aligns it to the WoRMS taxonomy.
    2. Generates the hierarchical curriculum datasets (maintains full network head).
    3. Generates the flat baseline datasets (for standard YOLO ablation studies).
    """
    print("=" * 60)
    print("🚀 Initiating End-to-End GFISHER Data Pipeline")
    print("=" * 60)

    # Step 1: Fetch, expand taxonomy, and align the datasets
    # This automatically categorizes them into 'train' and 'test' based on the URL strings
    print("\n--- Phase 1: Ingestion & WoRMS Alignment ---")
    expand_and_align_dataset(
        data_dir=DATA_DIR, 
        coco_sources=[TRAIN_DATA_URL, TEST_DATA_URL]
    )

    # 6. Convert to YOLO
    print("\nConverting aligned datasets to YOLO format...")
    # Filter out empty splits to prevent pycocowriter loops
    pycocowriter.coco2yolo.coco2yolo(DATA_DIR, DATA_DIR)


    # Step 2: Build the Hierarchical Curriculum 
    print("\n--- Phase 2: Hierarchical Curriculum Generation ---")
    build_hierarchical_curriculum(data_dir=DATA_DIR)

    # Step 3: Build the Flat Baselines 
    print("\n--- Phase 3: Flat Baseline Generation ---")
    build_flat_baselines(data_dir=DATA_DIR)

    print("\n" + "=" * 60)
    print("✅ GFISHER Pipeline Complete! All datasets and configs are ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
