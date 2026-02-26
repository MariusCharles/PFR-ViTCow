import os
from config import PRETRAIN_DIR
from cloud import download_sftp_pretrain_dataset
import argparse
from typing import List, Dict
import subprocess
import json

from tqdm import tqdm
from typing import Iterable, Optional

def progress_bar(iterable: Iterable,
                 desc: Optional[str] = None):
    return tqdm(
        iterable,
        desc=desc,
        unit="video",
        ncols=100,
        leave=True
    )

def get_num_frames(video_path: str) -> int:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return int(data["streams"][0]["nb_read_frames"])

def build_videomae_list(video_paths: List[str], out_file: str, desc: str):
    with open(out_file, "w") as f:
        for path in progress_bar(video_paths, desc=desc):
            n_frames = get_num_frames(path)
            f.write(f"{path} {n_frames}\n")

def get_train_test_local_paths() -> Dict[str, List[str]]:
    """
        Récupère les chemins locaux des vidéos train et test.

        Returns:
        Dict[str, List[str]]: {
            "train": [liste des chemins locaux des videos en train],
            "test": [liste des chemins locaux des videos en test]
        }
    """
    train_dir = os.path.abspath(f"{PRETRAIN_DIR}/train")
    test_dir = os.path.abspath(f"{PRETRAIN_DIR}/test")
    train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                    if f.endswith(".mp4")]
    test_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                    if f.endswith(".mp4")]
    return {
        "train": train_paths,
        "test": test_paths,
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", default=False,
                        help="Télécharger le dataset depuis le SFTP")
    args = parser.parse_args()

    if args.download:
        dataset = download_sftp_pretrain_dataset()
    else:
        dataset = get_train_test_local_paths()

    build_videomae_list(video_paths=dataset["train"], 
                        out_file=f"{PRETRAIN_DIR}/train.csv",
                        desc="Train csv creation")
    build_videomae_list(video_paths=dataset["test"], 
                        out_file=f"{PRETRAIN_DIR}/test.csv",
                        desc="Test csv creation")
