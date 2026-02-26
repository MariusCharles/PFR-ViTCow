import os
import shutil
import pandas as pd
import numpy as np
from config import TEST_FOLDER

# ======================
# PARAMETRES
# ======================
SOURCE_DIR = "pretraining_dataset/test"
CSV_PATH = "pretraining_dataset/test.csv"
OUTPUT_ROOT = "training_dataset"
VAL_RATIO = 0.2
SEED = 42
COPY_FILES = True  # False -> move
# ======================

np.random.seed(SEED)

# ----------------------
# LOAD CSV
# ----------------------
df = pd.read_csv(CSV_PATH, sep=";")
df.columns = ["filename", "label"]
df["split"] = "train"

# ----------------------
# SPLIT TEST (FIXE)
# ----------------------
df.loc[df["filename"].str.contains(TEST_FOLDER), "split"] = "test"

# ----------------------
# SPLIT TRAIN / VAL
# ----------------------
remaining = df[df["split"] != "test"]

for label, group in remaining.groupby("label"):
    indices = group.index.tolist()
    np.random.shuffle(indices)

    total = len(indices)
    max_val = int(np.floor(VAL_RATIO * total))
    if max_val < 1:
        max_val = 1

    val_indices = indices[:max_val]
    df.loc[val_indices, "split"] = "val"

# ----------------------
# CREATION ARBORESCENCE
# ----------------------
for _, row in df.iterrows():
    split = row["split"]
    label = row["label"]
    filename = row["filename"]

    label_dir = os.path.join(OUTPUT_ROOT, split, label)
    os.makedirs(label_dir, exist_ok=True)

    src = os.path.join(SOURCE_DIR, filename)
    dst = os.path.join(label_dir, filename)

    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

# ----------------------
# CREATION dataset.csv
# ----------------------
df["path"] = df.apply(
    lambda x: os.path.join(OUTPUT_ROOT, x["split"], x["label"], x["filename"]),
    axis=1
)

dataset_df = df[["path", "label", "split"]]
dataset_df.to_csv("dataset.csv", index=False)

# ----------------------
# ENCODAGE LABELS NUMERIQUES
# ----------------------
labels_sorted = sorted(df["label"].unique())
label_to_index = {label: idx for idx, label in enumerate(labels_sorted)}
df["label_index"] = df["label"].map(label_to_index)

# ----------------------
# EXPORT FORMAT VIDEOMAE
# ----------------------
def export_split(split_name):
    split_df = df[df["split"] == split_name]
    export_df = split_df[["path", "label_index"]]
    export_df.to_csv(
        f"{split_name}.csv",
        index=False,
        header=False,
        sep=" "
    )

export_split("train")
export_split("val")
export_split("test")

# ----------------------
# VERIFICATION
# ----------------------
print("Train:", len(df[df["split"] == "train"]))
print("Val:", len(df[df["split"] == "val"]))
print("Test:", len(df[df["split"] == "test"]))
print("\nValidation distribution:")
print(df[df["split"] == "val"]["label"].value_counts())
print("\nClass mapping:")
print(label_to_index)