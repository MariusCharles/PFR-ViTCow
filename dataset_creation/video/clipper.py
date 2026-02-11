import cv2
import os
import logging
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

def is_daytime_video(filepath: str, start: int, end: int) -> bool:
    """Check if video is in daytime hours based on filename timestamp.
    
    Verifies if the video falls within the daytime range (default 06:00 - 18:00)
    based on the 4 digits representing the hour in the filename.
    Works with full paths or just the filename.
    Example filename: 202502052200_D01.mp4
    
    Args:
        filepath (str): Path to the video file or filename.
        start (int): Start hour for daytime range (24-hour format).
        end (int): End hour for daytime range (24-hour format).
    
    Returns:
        bool: True if the video is within daytime hours, False otherwise.
    """
    filename = os.path.basename(filepath) 
    try:
        hour_str = filename[8:12] 
        hour = int(hour_str[:2])
        return start <= hour < end
    
    except Exception as e:
        logging.warning("Impossible de déterminer l'heure pour %s : %s", filename, e)
        return False


logger = logging.getLogger(__name__)

def extract_clips(
    video_path: str,
    output_folder: str,
    num_frames: int,
    step: int,
    num_clips: int,
    alias: str,
) -> None:
    """Extract a fixed number of clips randomly from a video.

    Args:
        video_path (str): Path to the input video.
        output_folder (str): Directory where output clips will be written.
        num_frames (int): Number of frames per output clip.
        step (int): Frame sampling step.
        num_clips (int): Number of clips to extract randomly.
        alias (str): Alias of the farm, used as prefix in clip filenames.
    
    Returns:
        None: Clips are written to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    name = os.path.basename(video_path)

    logger.info("Clipping video %s", name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.warning("Could not open video %s", name)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_start_idx = total_frames - num_frames * step
    if max_start_idx <= 0:
        logger.warning("Video %s too short for clip extraction", name)
        return

    # Choisir aléatoirement les indices de départ pour les clips parmi les indices possibles
    all_start_indices = list(range(0,max_start_idx + 1, num_frames * step))
    logger.info("Found %d possible clips for video %s", len(all_start_indices), name)
    start_indices = random.sample(all_start_indices, min(num_clips, len(all_start_indices)))
    #start_indices = random.sample(range(max_start_idx), min(num_clips, max_start_idx))

    pbar = tqdm(total=len(start_indices), desc=f"Clipping {name}", unit="clip")

    for start_idx in start_indices:
        clip_id = all_start_indices.index(start_idx) # Numéro du clip parmi tous les indices possibles
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(num_frames * step):
            ret, frame = cap.read()
            if not ret:
                break
            # Échantillonnage selon step
            if len(frames) < num_frames and (_ % step == 0):
                frames.append(frame)

        if frames:
            out_path = os.path.join(output_folder, f"{alias}_{name[:-4]}_clip{clip_id}.mp4")

            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / step,
                (w, h)
            )
            for f in frames:
                out.write(f)
            out.release()
        pbar.update(1)

    pbar.close()
    cap.release()
