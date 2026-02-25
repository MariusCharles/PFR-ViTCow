import os
import shutil
import subprocess

# -----------------------------
# Config
# -----------------------------
BASE_DIR = "/home/unruffled_satoshi/workdir/PFR-ViTCow/YOLO/subset_datasets_allfarms"
TEMP_DATASET = os.path.join(BASE_DIR, "dataset_yolo")  # temporary folder for training
YOLO_MODEL = "yolov8n-cls.pt"
EPOCHS = 150
IMGSZ = 224

# -----------------------------
# Find all train folders (exclude 'val')
# -----------------------------
all_folders = [d for d in os.listdir(BASE_DIR)
               if os.path.isdir(os.path.join(BASE_DIR, d)) and d != "val"]

VAL_FOLDER = os.path.join(BASE_DIR, "val")

for train_folder_name in all_folders:
    train_folder = os.path.join(BASE_DIR, train_folder_name)
    print(f"\n=== Training with: {train_folder_name} ===")

    # Clean temp folder
    if os.path.exists(TEMP_DATASET):
        shutil.rmtree(TEMP_DATASET)
    os.makedirs(TEMP_DATASET)

    # Copy validation folder
    val_dest = os.path.join(TEMP_DATASET, "val")
    shutil.copytree(VAL_FOLDER, val_dest)

    # Copy train folder
    train_dest = os.path.join(TEMP_DATASET, "train")
    shutil.copytree(train_folder, train_dest)

    print(f"Dataset prepared at {TEMP_DATASET}")

    # Run YOLOv8 classify training using folder directly
    output_name = f"yolo_{train_folder_name}"
    command = [
        "yolo", "classify", "train",
        f"model={YOLO_MODEL}",
        f"data={TEMP_DATASET}", 
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"name={output_name}"
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)

    # Cleanup temp folder
    shutil.rmtree(TEMP_DATASET)
    print(f"Finished training for {train_folder_name}, temp folder removed.\n")
