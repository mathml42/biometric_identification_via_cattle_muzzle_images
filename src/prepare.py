# src/prepare.py
# Assisted by Gemini [cite: 37]

import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    raw_dir = Path(params["prepare"]["raw_data_dir"])
    prep_dir = Path(params["prepare"]["prepared_data_dir"])
    test_size = params["prepare"]["test_size"]
    min_images = params["prepare"]["min_images"]
    random_state = params["prepare"]["random_state"]

    # Create output directories for the split
    train_dir = prep_dir / "train"
    test_dir = prep_dir / "test"
    
    # Clean up existing prepared data if it exists to ensure reproducibility
    if prep_dir.exists():
        shutil.rmtree(prep_dir)
        
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    all_labels = []

    # 1. Filter and Collect Data
    for label_folder in raw_dir.iterdir():
        if not label_folder.is_dir():
            continue
            
        label = label_folder.name
        images = list(label_folder.glob("*.*")) # Captures .jpg, .png, etc.

        # Rule: Ignore folders that contain fewer than 3 images 
        if len(images) < min_images:
            print(f"Skipping label '{label}' - only {len(images)} images found.")
            continue

        for img in images:
            all_images.append(img)
            all_labels.append(label)

    print(f"\nTotal valid images: {len(all_images)}")
    print(f"Total valid classes: {len(set(all_labels))}")

    # 2. The "Golden" Split (90% Train/Val, 10% Test) 
    # Note: We don't use 'stratify=all_labels' here because some classes 
    # might have exactly 3 images, making a strict 10% stratified split mathematically impossible.
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, 
        all_labels, 
        test_size=test_size, 
        random_state=random_state
    )

    # 3. Copy files to the prepared directory
    def copy_files(files, labels, split_dir):
        for file_path, label in zip(files, labels):
            if label == "Cattle photos":
                continue
            dest_folder = split_dir / label
            dest_folder.mkdir(exist_ok=True)
            shutil.copy(file_path, dest_folder / file_path.name)

    copy_files(X_train, y_train, train_dir)
    copy_files(X_test, y_test, test_dir)

    print(f"\nData successfully split and saved to {prep_dir}")
    print(f"Training/Validation samples: {len(X_train)}")
    print(f"Untouched Test samples: {len(X_test)}")

if __name__ == "__main__":
    main()