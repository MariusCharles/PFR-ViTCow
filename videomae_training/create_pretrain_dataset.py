import os
from config import NUM_FRAMES_PER_CLIP, PRETRAIN_DIR
from pipeline.cloud import download_sftp_pretrain_dataset
import argparse
from typing import List, Dict

def build_videomae_list(video_paths: List[str], out_file: str):
    with open(out_file, "w") as f:
        for path in video_paths:
            n_frames = NUM_FRAMES_PER_CLIP
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
                        out_file=f"{PRETRAIN_DIR}/train.csv")
    build_videomae_list(video_paths=dataset["test"], 
                        out_file=f"{PRETRAIN_DIR}/test.csv")
