from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score
)

# ====== CONFIG ======
MODEL_PATH = "runs/classify/yolo_size_1000_split_01/weights/best.pt"
SOURCE_PATH = "/home/unruffled_satoshi/workdir/PFR-ViTCow/YOLO/validation_set"
ANNOTATIONS_CSV = "all_annotations.csv"
# ====================


# -----------------------------
# Load model and predict
# -----------------------------
model = YOLO(MODEL_PATH)
results = model.predict(source=SOURCE_PATH, stream=True, verbose=False)

rows = []
class_names = model.names

for r in results:
    probs = r.probs.data.cpu().numpy()
    row = {"image": Path(r.path).name}

    for i, class_name in class_names.items():
        row[class_name] = float(probs[i])

    row["predicted_class"] = class_names[int(r.probs.top1)]
    rows.append(row)

df_preds = pd.DataFrame(rows)


# -----------------------------
# Frame → video conversion
# -----------------------------
def frame_to_video(frame_name):
    return frame_name.rsplit("_", 1)[0] + ".mp4"

df_preds["video_file"] = df_preds["image"].apply(frame_to_video)


# -----------------------------
# Aggregate per video (majority vote)
# -----------------------------
video_rows = []

for video, group in df_preds.groupby("video_file"):
    majority_class = Counter(group["predicted_class"]).most_common(1)[0][0]
    
    mean_probs = group[class_names.values()].mean().to_dict()
    row = {"video_file": video, "predicted_class": majority_class}
    row.update(mean_probs)
    video_rows.append(row)

df_video = pd.DataFrame(video_rows)


# -----------------------------
# Load ground truth
# -----------------------------
df_ann = pd.read_csv(ANNOTATIONS_CSV, header=None, names=["video_file", "true_class"])


# -----------------------------
# Merge
# -----------------------------
df = df_video.merge(df_ann, on="video_file", how="inner")


# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([df["true_class"], df["predicted_class"]]))

df["y_true"] = label_encoder.transform(df["true_class"])
df["y_pred"] = label_encoder.transform(df["predicted_class"])

class_order = label_encoder.classes_
y_scores = df[class_order].values

y_true = df["y_true"].values
y_pred = df["y_pred"].values


# -----------------------------
# Compute metrics
# -----------------------------
y_true_onehot = np.zeros_like(y_scores)
y_true_onehot[np.arange(len(y_true)), y_true] = 1

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
    "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
    "f1_micro": fbeta_score(y_true, y_pred, beta=1.0, average="micro", zero_division=0),
    
    "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "f1_macro": fbeta_score(y_true, y_pred, beta=1.0, average="macro", zero_division=0),

    "mAP_macro": average_precision_score(y_true_onehot, y_scores, average="macro"),
    "mAP_micro": average_precision_score(y_true_onehot, y_scores, average="micro"),
}

print(metrics)