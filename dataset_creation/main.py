import os
import subprocess
import logging
import config
from video.clipper import extract_clips, is_daytime_video
from pipeline.cloud import list_sftp_videos, download_sftp_video, remove_video, upload_video, check_if_exists
import json
import random

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("Pipeline started")

def clean_mp4_files(folder_path: str):
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith(".mp4"):
                file_path = os.path.join(folder_path, f)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Erreur lors de la suppression de {file_path}: {e}")

clean_mp4_files(config.CLIP_FOLDER)
clean_mp4_files(config.CROP_FOLDER)

videos_path = list_sftp_videos(config.FARM_NAMES)

# Sauvegarder videos_path avant filtrage
with open("videos_before_filter.json", "w") as f:
    json.dump(videos_path, f, indent=2)

videos_path = [path for path in videos_path if is_daytime_video(path.get("filename", ""), config.START, config.END)]

# Sauvegarder videos_path après filtrage
with open("videos_after_filter.json", "w") as f:
    json.dump(videos_path, f, indent=2)

random.shuffle(videos_path)

for iteration in range(1000):
    video_path = random.choice(videos_path)
    alias = video_path["alias"]
    
    local_path = download_sftp_video(video_path["filename"], alias)

    extract_clips(
        local_path,
        config.CLIP_FOLDER,
        config.NUM_FRAMES_PER_CLIP,
        config.FRAME_STEP,
        config.NUM_CLIP,
        alias,
    )

    remove_video(local_path)

    clips_to_process = [clip for clip in sorted(os.listdir(config.CLIP_FOLDER))]
    processed_count = 0

    for clip in clips_to_process:
        if not clip.lower().endswith(".mp4"):
            continue

        # Vérifier si le clip a déjà été traité
        if check_if_exists(clip):
            logging.info("Clip %s already processed, skipping.", clip)
            continue

        processed_count += 1

        clip_path = os.path.join(config.CLIP_FOLDER, clip)
        logging.info("Processing clip %s", clip_path)

        try:
            result = subprocess.run(
                ["python", "process_clip.py", clip_path],
                capture_output=True,
                text=True,
                check=True
            )

            # Prendre uniquement la dernière ligne du stdout pour le JSON
            json_line = result.stdout.strip().splitlines()[-1]
            out_all_paths = json.loads(json_line)

            logging.info("out_all_paths: %s", out_all_paths)
            for out_path in out_all_paths:
                upload_video(out_path, alias)
                remove_video(out_path)

        except subprocess.CalledProcessError as e:
            logging.error("Clip processing failed: %s", clip_path)
            logging.error("STDOUT: %s", e.stdout)
            logging.error("STDERR: %s", e.stderr)
            continue  # passe au clip suivant

        finally:
            # toujours supprimer le clip même si erreur
            remove_video(clip_path)

    logging.info("Finished video %s", os.path.basename(local_path))
    logging.info("Processing complete: %d clips processed, %d clips skipped.",
             processed_count,len(clips_to_process) - processed_count)

logging.info("Pipeline finished")
