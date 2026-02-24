import os
import pandas as pd
import subprocess
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFTP_USER, SFTP_HOST, SFTP_PORT, UPLOAD_DIR

# =========================
# CONFIG
# =========================

ANNOTATIONS_FILE = "annotations.csv"
LOCAL_DIR = "local_splits"  # Local directory for file generation
SPLIT_DIR = LOCAL_DIR
GENERATED_DIR = os.path.join(LOCAL_DIR, "generated")

# Remote SFTP paths
REMOTE_EVAL_DIR = f"{UPLOAD_DIR}/evaluation"
REMOTE_SPLIT_DIR = REMOTE_EVAL_DIR

TEST_FARM = "BUISSON"  # Videos from this farm go to test set
RANDOM_STATE = 42

TRAIN_SIZES = [1000, 1500, 2000, 3000]   # ← modifiable librement
N_SPLITS = 3                            # repeated subsampling

# =========================
# HELPERS
# =========================

def ensure_dirs():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

def save_list(series, filepath):
    series.to_csv(filepath, index=False, header=False)

def upload_file_sftp(local_path, remote_path):
    """Upload a file to SFTP server."""
    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]
    
    # Create remote directory if needed
    remote_dir = os.path.dirname(remote_path)
    mkdir_commands = f"mkdir {remote_dir}\n"
    
    # Upload file
    upload_command = f'put "{local_path}" "{remote_path}"\n'
    
    proc = subprocess.run(
        cmd,
        input=mkdir_commands + upload_command + "exit\n",
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        print(f"  Warning: Upload may have failed for {os.path.basename(local_path)}")
    else:
        print(f"  ✓ Uploaded: {os.path.basename(local_path)}")

def upload_directory_sftp(local_dir, remote_dir):
    """Upload all files in a directory to SFTP server, maintaining structure."""
    print(f"\nUploading to SFTP: {remote_dir}")
    
    # Create base remote directory
    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]
    subprocess.run(
        cmd,
        input=f"mkdir {remote_dir}\nexit\n",
        capture_output=True,
        text=True
    )
    
    # Upload all files
    for root, dirs, files in os.walk(local_dir):
        # Get relative path from local_dir
        rel_path = os.path.relpath(root, local_dir)
        
        # Create remote subdirectories
        if rel_path != ".":
            remote_subdir = os.path.join(remote_dir, rel_path).replace("\\", "/")
            subprocess.run(
                cmd,
                input=f"mkdir {remote_subdir}\nexit\n",
                capture_output=True,
                text=True
            )
        
        # Upload files in this directory
        for file in files:
            local_file = os.path.join(root, file)
            if rel_path == ".":
                remote_file = f"{remote_dir}/{file}"
            else:
                remote_file = os.path.join(remote_dir, rel_path, file).replace("\\", "/")
            
            upload_file_sftp(local_file, remote_file)

# =========================
# STEP 1 — INIT SPLIT (IMMUTABLE)
# =========================

def init_train_test_split(df):
    """
    Create train/test split based on farm.
    BUISSON videos go to test, all others go to train.
    Keep class balance in both sets.
    """
    print(f"Creating FIXED train/test split (test farm: {TEST_FARM})...")

    # Split by farm
    test_df = df[df["Video"].str.contains(TEST_FARM, na=False)]
    train_df = df[~df["Video"].str.contains(TEST_FARM, na=False)]

    save_list(test_df["Video"], os.path.join(SPLIT_DIR, "test.txt"))
    save_list(train_df["Video"], os.path.join(SPLIT_DIR, "train_full.txt"))

    print("Saved:")
    print(f"  TEST ({TEST_FARM}) → {len(test_df)} samples")
    print(f"  TRAIN_FULL (others) → {len(train_df)} samples")
    
    # Show class distribution
    print("\nClass distribution in test set:")
    print(test_df["Label"].value_counts().sort_index())
    print("\nClass distribution in train set:")
    print(train_df["Label"].value_counts().sort_index())
    
    print(f"\n  Location: {SPLIT_DIR}")

    return train_df, test_df

# =========================
# STEP 2 — GENERATE SUBSETS
# =========================

def generate_train_subsets(train_df):
    print("\nGenerating TRAIN subsets...")

    X = train_df["Video"].values
    y = train_df["Label"].values

    for size in TRAIN_SIZES:

        if size >= len(train_df):
            print(f"Skipping size {size} (>= train_full)")
            continue

        size_dir = os.path.join(GENERATED_DIR, f"size_{size}")
        os.makedirs(size_dir, exist_ok=True)

        print(f"\nSize {size}:")

        sss = StratifiedShuffleSplit(
            n_splits=N_SPLITS,
            train_size=size,
            random_state=RANDOM_STATE
        )

        for split_id, (indices, _) in enumerate(sss.split(X, y), start=1):

            subset_videos = X[indices]

            filepath = os.path.join(size_dir, f"split_{split_id:02d}.txt")

            pd.Series(subset_videos).to_csv(
                filepath,
                index=False,
                header=False
            )

        print(f"  Generated {N_SPLITS} splits")

# =========================
# MAIN
# =========================

def main():

    ensure_dirs()

    df = pd.read_csv(ANNOTATIONS_FILE)

    if not {"Video", "Label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: Video, Label")

    test_path = os.path.join(SPLIT_DIR, "test.txt")
    train_full_path = os.path.join(SPLIT_DIR, "train_full.txt")

    if not os.path.exists(test_path):
        train_df, test_df = init_train_test_split(df)
    else:
        print("Using existing FIXED split.")
        train_videos = pd.read_csv(train_full_path, header=None)[0]
        train_df = df[df["Video"].isin(train_videos)]

    generate_train_subsets(train_df)

    # Upload all files to SFTP
    print("\n" + "="*60)
    print("UPLOADING TO SFTP SERVER")
    print("="*60)
    upload_directory_sftp(SPLIT_DIR, REMOTE_SPLIT_DIR)
    
    print("\n" + "="*60)
    print("DONE - All files uploaded to cloud")
    print("="*60)

if __name__ == "__main__":
    main()