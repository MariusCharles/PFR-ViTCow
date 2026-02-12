import subprocess
import os
import logging
import re
from typing import Dict, List

# Import SFTP server params 
from config import (
    SFTP_USER,
    SFTP_HOST,
    SFTP_PORT,
    UPLOAD_DIR,
    FARM_NAMES,
    TEST_FOLDER,
    PRETRAIN_DIR
)


def list_sftp_videos(folders_dict: Dict[str, str]) -> List[Dict[str, str]]:
    """Lists all video files from specified SFTP server folders.
    
    Args:
        folders_dict (Dict[str, str]): Dictionary with alias as key and remote path as value.
    
    Returns:
        List[Dict[str, str]]: List of dicts with 'filename' and 'alias' for each video found.
    """
    folder_pattern = re.compile(r"^D\d{2}$")
    video_pattern = re.compile(r"\.mp4$")

    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]

    all_videos = []

    for alias, remote_path in folders_dict.items():
        # First, list the farm directory to find DXX folders
        proc = subprocess.run(
            cmd,
            input=f"ls \"{remote_path}\"\nexit\n",
            capture_output=True,
            text=True
        )

        subfolders = []
        for line in proc.stdout.splitlines():
            line = line.replace("sftp>", "").strip()
            if not line:
                continue
            name = os.path.basename(line)
            if folder_pattern.match(name):
                subfolders.append(name)

        # Then, for each DXX subfolder, list videos
        for subfolder in subfolders:
            subfolder_path = f"{remote_path.rstrip('/')}/{subfolder}"
            proc_sub = subprocess.run(
                cmd,
                input=f"ls \"{subfolder_path}\"\nexit\n",
                capture_output=True,
                text=True
            )

            for line in proc_sub.stdout.splitlines():
                line = line.replace("sftp>", "").strip()
                if not line:
                    continue
                if video_pattern.search(line):
                    all_videos.append({'filename': f"{subfolder}/{os.path.basename(line)}", 'alias': alias})

    return all_videos


def download_sftp_video(filename: str, alias: str, local_dir: str) -> str:
    """Downloads a video file from the SFTP server.
    
    Args:
        filename (str): The name of the video file.
        alias (str): The alias of the folder where the video is located.
        local_dir (str): The path to the local directory where the videos are to be downloaded.
    
    Returns:
        str: The local path where the video was downloaded.
    """
    remote_path = f"{FARM_NAMES[alias]}/{filename}"
    logging.info("Downloading video %s from %s", filename, remote_path)
    local_path = os.path.join(local_dir, os.path.basename(filename))

    os.makedirs(local_dir, exist_ok=True)

    cmd = f"echo 'get \"{remote_path}\" \"{local_path}\"' | sftp -P {SFTP_PORT} {SFTP_USER}@{SFTP_HOST}"
    subprocess.run(cmd, shell=True, check=True)
    return local_path


def download_sftp_pretrain_dataset() -> Dict[str, List[str]]:
    """
    Télécharge le dataset de prétraining depuis UPLOAD_DIR.

    - Tous les fichiers vidéo présents dans tous les sous-dossiers de UPLOAD_DIR
      sauf TEST_FOLDER -> dataset/train
    - Tous les fichiers vidéo dans TEST_FOLDER -> dataset/test

    Returns:
        Dict[str, List[str]]: {
            "train": [liste des chemins locaux des videos en train],
            "test": [liste des chemins locaux des videos en test]
        }
    """
    cmd = ["sftp", "-P", str(SFTP_PORT), f"{SFTP_USER}@{SFTP_HOST}"]
    proc = subprocess.run(
        cmd,
        input=f'ls "{UPLOAD_DIR}"\nexit\n',
        capture_output=True,
        text=True
    )

    folders = [os.path.basename(line.replace("sftp>", "").strip())
               for line in proc.stdout.splitlines() if line.strip()]
    folders_dict = {folder: f"{UPLOAD_DIR}/{folder}" for folder in folders}
    all_videos = list_sftp_videos(folders_dict)

    train_dir = os.path.join(PRETRAIN_DIR, "train")
    test_dir = os.path.join(PRETRAIN_DIR, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_files = []
    test_files = []
    total = len(all_videos)

    for i,video in enumerate(all_videos):
        alias = video["alias"]
        filename = video["filename"]

        
        
        if alias == TEST_FOLDER:
            local_path = os.path.join(test_dir,os.path.basename(filename))
            download_sftp_video(filename, alias, test_dir)
            test_files.append(local_path)
        else:
            local_path = os.path.join(train_dir,os.path.basename(filename))
            download_sftp_video(filename, alias, train_dir)
            train_files.append(local_path)

        progress = int((i / total) * 50)
        bar = "#" * progress + "-" * (50 - progress)
        print(f"\rTéléchargement: [{bar}] {i}/{total} videos", end="", flush=True)
    print("\nTéléchargement terminé")

    return {
        "train": train_files,
        "test": test_files,
    }
