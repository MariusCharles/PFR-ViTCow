"""
Efficiency Analysis: Train linear classifier on embeddings for different dataset sizes.

Usage:
    python efficiency_analysis.py --dataset subset_1000_split_01
    python efficiency_analysis.py --dataset subset_1500_split_02
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFTP_USER, SFTP_HOST, SFTP_PORT, UPLOAD_DIR
from evaluation.metrics import classification_metrics

# =========================
# CONSTANTS
# =========================

EMBEDDINGS_FILE = "all_embeddings.json"
REMOTE_EMBEDDINGS = f"{UPLOAD_DIR}/zero_shot_results/final_pretrain/all_embeddings.json"
EVALUATION_DIR = "evaluation"
LOCAL_SPLITS = "local_splits"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 8  # Will be computed from annotations.csv
EMBEDDING_DIM = 1536  # VideoMAE embeddings dimension

# Training params
BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# =========================
# HELPERS
# =========================

def download_from_sftp(remote_path: str, local_path: str) -> None:
    """Download file from SFTP server."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]
    
    proc = subprocess.run(
        cmd,
        input=f'get "{remote_path}" "{local_path}"\nexit\n',
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        raise Exception(f"Failed to download {remote_path}: {proc.stderr}")
    
    print(f"✓ Downloaded: {local_path}")


def download_evaluation_splits() -> None:
    """Download evaluation splits from SFTP if not present locally."""
    if os.path.exists(LOCAL_SPLITS):
        print(f"✓ {LOCAL_SPLITS}/ already exists locally")
        return
    
    print(f"Downloading evaluation splits from SFTP...")
    remote_eval = f"{UPLOAD_DIR}/evaluation"
    
    # Create evaluation directory and download all files
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    # Download files recursively using SFTP
    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]
    
    # This is a simplified approach 
    proc = subprocess.run(
        cmd,
        input=f'get -r "{remote_eval}" "{EVALUATION_DIR}"\nexit\n',
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        print(f"Warning: Recursive download may have issues: {proc.stderr}")
    
    # Rename evaluation to local_splits
    if os.path.exists(EVALUATION_DIR):
        if os.path.exists(LOCAL_SPLITS):
            import shutil
            shutil.rmtree(LOCAL_SPLITS)
        os.rename(EVALUATION_DIR, LOCAL_SPLITS)
    
    print(f"✓ Downloaded evaluation splits")


def ensure_embeddings() -> None:
    """Download all_embeddings.json if not present."""
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"✓ {EMBEDDINGS_FILE} already exists")
        return
    
    print(f"Downloading {EMBEDDINGS_FILE}...")
    download_from_sftp(REMOTE_EMBEDDINGS, EMBEDDINGS_FILE)


def load_embeddings_json() -> Dict:
    """Load all embeddings from JSON file."""
    with open(EMBEDDINGS_FILE, 'r') as f:
        data = json.load(f)
    return data


def parse_dataset_name(dataset_name: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse dataset name like 'subset_1000_split_01' or 'train_full'
    Returns: (dataset_type, size, split_id)
    """
    if dataset_name == 'train_full':
        return 'train_full', None, None
    
    parts = dataset_name.split('_')
    if len(parts) >= 4:  # subset, 1000, split, 01
        size = int(parts[1])
        split_id = int(parts[3])
        return 'subset', size, split_id
    else:
        raise ValueError(f"Invalid dataset name format: {dataset_name}")


def load_video_list(filepath: str) -> List[str]:
    """Load list of video names from .txt file."""
    with open(filepath, 'r') as f:
        videos = [line.strip() for line in f if line.strip()]
    return videos


def extract_video_name(path: str) -> str:
    """Extract video name from full path in embeddings JSON."""
    return Path(path).name


def prepare_data(
    embeddings_data: Dict,
    train_videos: List[str],
    test_videos: List[str],
    annotations_df
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare training and test data from embeddings.
    Uses annotations.csv as source of truth for labels.
    
    Returns:
        X_train, y_train, X_test, y_test, label_encoder_dict
    """
    # Create video name to embedding mapping from embeddings.json
    video_to_embedding = {}
    
    for entry in embeddings_data:
        video_name = extract_video_name(entry['path'])
        video_to_embedding[video_name] = np.array(entry['embedding'], dtype=np.float32)
    
    # Create video name to label mapping from annotations.csv
    video_to_label = {}
    label_set = set()
    
    for _, row in annotations_df.iterrows():
        video_name = row['Video']
        label = row['Label']
        video_to_label[video_name] = label
        label_set.add(label)
    
    # Encode labels
    le = LabelEncoder()
    label_names = sorted(list(label_set))
    le.fit(label_names)
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    
    # Extract train embeddings and labels
    X_train_list = []
    y_train_list = []
    
    for video in train_videos:
        if video in video_to_embedding:
            X_train_list.append(video_to_embedding[video])
            y_train_list.append(label_to_idx[video_to_label[video]])
        else:
            print(f"Warning: {video} not found in embeddings")
    
    # Extract test embeddings and labels
    X_test_list = []
    y_test_list = []
    
    for video in test_videos:
        if video in video_to_embedding:
            X_test_list.append(video_to_embedding[video])
            y_test_list.append(label_to_idx[video_to_label[video]])
        else:
            print(f"Warning: {video} not found in embeddings")
    
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)
    
    return X_train, y_train, X_test, y_test, {
        'label_names': label_names,
        'label_to_idx': label_to_idx
    }


def train_linear_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    embedding_dim: int
) -> Tuple[torch.nn.Module, Dict]:
    """
    Train linear classifier on embeddings.
    
    Returns:
        model, predictions_dict
    """
    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Compute class weights for imbalanced data
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = len(y_train) / (num_classes * counts)
    class_weights = torch.from_numpy(class_weights).float().to(DEVICE)
    
    print(f"  Class weights: {dict(zip(range(num_classes), class_weights.cpu().numpy()))}")
    
    # Create model
    model = nn.Linear(embedding_dim, num_classes)
    model.to(DEVICE)
    
    # Use weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    print(f"Training for {NUM_EPOCHS} epochs...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
        
        avg_loss = running_loss / len(train_dataset)
        if (epoch + 1) % max(1, NUM_EPOCHS // 5) == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Evaluate
    print("Evaluating on test set...")
    model.eval()
    
    y_pred_list = []
    y_scores_list = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            
            y_pred_list.append(outputs.argmax(dim=1).cpu().numpy())
            y_scores_list.append(probs.cpu().numpy())
    
    y_pred = np.concatenate(y_pred_list)
    y_scores = np.concatenate(y_scores_list)
    
    return model, {
        'y_pred': y_pred,
        'y_scores': y_scores,
        'y_test': y_test
    }


def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    num_classes: int = None
) -> Dict:
    """Compute all metrics."""
    from sklearn.preprocessing import label_binarize
    
    metrics = classification_metrics(
        y_test,
        y_pred,
        y_scores=None,  # Don't pass y_scores with multiclass - we'll handle mAP separately
        beta=1.0,
        average_type="macro"
    )
    
    # For multiclass mAP, we need to binarize the labels
    if num_classes and num_classes > 2:
        try:
            y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
            from sklearn.metrics import average_precision_score
            metrics["mAP_macro"] = average_precision_score(
                y_test_bin, y_scores, average="macro"
            )
            metrics["mAP_micro"] = average_precision_score(
                y_test_bin, y_scores, average="micro"
            )
        except Exception as e:
            print(f"Warning: Could not compute mAP: {e}")
    
    metrics['num_train'] = None  # Will be set by caller
    metrics['num_test'] = len(y_test)
    
    return metrics


def save_results(results: Dict, output_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")


def upload_results_to_sftp(local_path: str, dataset_name: str) -> None:
    """Upload results to SFTP server."""
    remote_dir = f"{UPLOAD_DIR}/evaluation/results"
    remote_file = f"{remote_dir}/{dataset_name}.json"
    
    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]
    
    # Create remote directory
    subprocess.run(
        cmd,
        input=f"mkdir {remote_dir}\nexit\n",
        capture_output=True,
        text=True
    )
    
    # Upload file
    proc = subprocess.run(
        cmd,
        input=f'put "{local_path}" "{remote_file}"\nexit\n',
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        print(f"Warning: Upload may have failed: {proc.stderr}")
    else:
        print(f"✓ Uploaded results to: {remote_file}")


def main():
    parser = argparse.ArgumentParser(description='Efficiency Analysis')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., subset_1000_split_01)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EFFICIENCY ANALYSIS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {DEVICE}")
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Num classes: {NUM_CLASSES}")
    
    # Step 1: Ensure we have embeddings
    print("\n[1/5] Checking embeddings...")
    ensure_embeddings()
    
    # Step 2: Ensure we have evaluation splits
    print("\n[2/5] Checking evaluation splits...")
    download_evaluation_splits()
    
    # Step 3: Load data
    print("\n[3/5] Loading data...")
    subset_dir, subset_size, split_id = parse_dataset_name(args.dataset)
    
    # Handle train_full special case
    if subset_dir == 'train_full':
        train_list_path = os.path.join(LOCAL_SPLITS, "train_full.txt")
    else:
        train_list_path = os.path.join(
            LOCAL_SPLITS, "generated", f"size_{subset_size}", f"split_{split_id:02d}.txt"
        )
    
    test_list_path = os.path.join(LOCAL_SPLITS, "test.txt")
    
    if not os.path.exists(train_list_path):
        raise FileNotFoundError(f"Train list not found: {train_list_path}")
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Test list not found: {test_list_path}")
    
    train_videos = load_video_list(train_list_path)
    test_videos = load_video_list(test_list_path)
    
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Test videos: {len(test_videos)}")
    
    # Load embeddings
    embeddings_data = load_embeddings_json()
    print(f"  Total embeddings: {len(embeddings_data)}")
    
    # Load annotations.csv for labels
    import pandas as pd
    annotations_df = pd.read_csv("annotations.csv", sep=",")
    # Strip whitespace from column names
    annotations_df.columns = annotations_df.columns.str.strip()
    print(f"  Annotations loaded: {len(annotations_df)} entries")
    print(f"  Columns: {list(annotations_df.columns)}")
    
    # Prepare data
    X_train, y_train, X_test, y_test, label_info = prepare_data(
        embeddings_data, train_videos, test_videos, annotations_df
    )
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Classes: {label_info['label_names']}")
    
    # Step 4: Train linear classifier
    print("\n[4/5] Training linear classifier...")
    num_classes = len(label_info['label_names'])
    print(f"  Num classes: {num_classes}")
    
    model, predictions = train_linear_classifier(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Step 5: Compute metrics and save
    print("\n[5/5] Computing metrics and saving results...")
    metrics = compute_metrics(
        predictions['y_test'],
        predictions['y_pred'],
        predictions['y_scores'],
        num_classes=num_classes
    )
    metrics['num_train'] = len(train_videos)
    metrics['subset_size'] = subset_size
    metrics['split_id'] = split_id
    metrics['dataset_name'] = args.dataset
    
    # Save locally
    os.makedirs(RESULTS_DIR, exist_ok=True)
    local_result_path = os.path.join(RESULTS_DIR, f"{args.dataset}.json")
    save_results(metrics, local_result_path)
    
    # Upload to SFTP
    upload_results_to_sftp(local_result_path, args.dataset)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print(f"\nResults Summary:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    f_beta_key = 'f1.0'
    if f_beta_key in metrics:
        print(f"  F1: {metrics[f_beta_key]:.4f}")
    if 'mAP_macro' in metrics:
        print(f"  mAP (macro): {metrics['mAP_macro']:.4f}")
        print(f"  mAP (micro): {metrics['mAP_micro']:.4f}")


if __name__ == '__main__':
    main()
