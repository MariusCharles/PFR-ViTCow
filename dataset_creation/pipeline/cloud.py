import subprocess
import os
import logging
import config
import re
from typing import Dict, List

# Import SFTP server params 
from config import (
    SFTP_USER,
    SFTP_HOST,
    SFTP_PORT,
    REMOTE_DIR,
    UPLOAD_DIR,
    LOCAL_TMP_DIR,
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


def download_sftp_video(filename: str, alias: str) -> str:
    """Downloads a video file from the SFTP server.
    
    Args:
        filename (str): The name of the video file.
        alias (str): The alias of the folder where the video is located.
    
    Returns:
        str: The local path where the video was downloaded.
    """
    remote_path = f"{FARM_NAMES[alias]}/{filename}"
    logging.info("Downloading video %s from %s", filename, remote_path)
    local_path = os.path.join(LOCAL_TMP_DIR, os.path.basename(filename))

    os.makedirs(LOCAL_TMP_DIR, exist_ok=True)

    cmd = f"echo 'get \"{remote_path}\" \"{local_path}\"' | sftp -P {SFTP_PORT} {SFTP_USER}@{SFTP_HOST}"
    subprocess.run(cmd, shell=True, check=True)
    return local_path

def remove_video(local_path: str) -> None:
    """Removes a video file if it exists.
    
    Args:
        local_path (str): The local path of the video file to remove.
    """
    if os.path.exists(local_path):
        os.remove(local_path)


CREATED_FOLDERS = set()

def mkdir_sftp(remote_dir: str) -> None:
    if remote_dir in CREATED_FOLDERS:
        return
    
    folders = remote_dir.strip("/").split("/")
    cmds = []
    current_path = ""
    for folder in folders:
        current_path += f"/{folder}"
        cmds.append(f"-mkdir \"{current_path}\"")
    
    batch_cmds = "\n".join(cmds) + "\nexit"
    cmd = f"echo '{batch_cmds}' | sftp -b - -P {SFTP_PORT} {SFTP_USER}@{SFTP_HOST}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info("Dossier %s prêt (créé ou existant) ", remote_dir)
        CREATED_FOLDERS.add(remote_dir)
    else:
        logging.error("Erreur de création du dossier %s (Code %d) : %s", 
                      remote_dir, result.returncode, result.stderr)
        


def upload_video(local_path: str, alias: str) -> str:
    """Uploads a video file to the SFTP server with a prefix in the filename.
    
    Args:
        local_path (str): The local path of the video file to upload.
        alias (str): The alias of the farm, used as prefix for the filename.
    
    Returns:
        str: The local path of the uploaded video.
    """
    original_filename = os.path.basename(local_path)
    
    remote_dir_full = f"{UPLOAD_DIR}/{alias}"
    logging.info("Uploading video %s to %s", original_filename, remote_dir_full)
    
    mkdir_sftp(remote_dir_full)

    put_cmd = f"echo 'put \"{local_path}\" \"{remote_dir_full}/{original_filename}\"\nexit' | sftp -P {SFTP_PORT} {SFTP_USER}@{SFTP_HOST}"
    subprocess.run(put_cmd, shell=True, check=True)

    return local_path


def check_if_exists(local_path: str) -> bool:
    """
    Vérifie récursivement si un fichier contenant local_path existe sur le serveur.
    
    Args:
        local_path (str): Le nom du clip à rechercher.
        
    Returns:
        bool: True si le fichier existe, False sinon.
    """
    find_cmd = (
        f"ssh -p {SFTP_PORT} {SFTP_USER}@{SFTP_HOST} "
        f"\"find {UPLOAD_DIR} -type f -name '*{local_path}*'\""
    )

    result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
    found_files = result.stdout.splitlines()

    pattern = re.compile(re.escape(local_path))  # échappe les caractères spéciaux

    try: 
        for f in found_files:
            basename = os.path.basename(f)
            if pattern.search(basename):
                logging.info(f"Fichier correspondant à '{local_path}' déjà présent sur le cloud : {f}")
                return True
        return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de la vérification sur le cloud : {e}")
        return False

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

        local_path = os.path.join(
            test_dir if alias == TEST_FOLDER else train_dir,
            os.path.basename(filename)
        )
        download_sftp_video(filename, alias)
        
        if alias == TEST_FOLDER:
            test_files.append(local_path)
        else:
            train_files.append(local_path)

        progress = int((i / total) * 50)
        bar = "#" * progress + "-" * (50 - progress)
        print(f"\rTéléchargement: [{bar}] {i}/{total} videos", end="", flush=True)
    print("\nTéléchargement terminé")

    return {
        "train": train_files,
        "test": test_files,
    }
