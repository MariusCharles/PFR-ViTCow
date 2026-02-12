import cv2
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional
import config

logger = logging.getLogger(__name__)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask to a bounding box.
    
    Args:
        mask (np.ndarray): Binary mask array.
    
    Returns:
        Optional[Tuple[int, int, int, int]]: Bounding box as (x1, y1, x2, y2) or None if empty.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_size(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Calculate the width and height of a bounding box.
    
    Args:
        bbox (Tuple[int, int, int, int]): Bounding box as (x1, y1, x2, y2).
    
    Returns:
        Tuple[int, int]: Width and height as (w, h).
    """
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def make_square_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert a bounding box to a square bounding box.
    
    Args:
        bbox (Tuple[int, int, int, int]): Bounding box as (x1, y1, x2, y2).
    
    Returns:
        Tuple[int, int, int, int]: Square bounding box as (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    w, h = bbox_size(bbox)
    side = max(w, h)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = side // 2

    return (
        cx - half,
        cy - half,
        cx + half,
        cy + half,
    )


def crop_with_clamp(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_size: int,
) -> np.ndarray:
    """Crop a frame with bounding box and pad with zeros if needed.
    
    Args:
        frame (np.ndarray): Input frame to crop.
        bbox (Tuple[int, int, int, int]): Bounding box as (x1, y1, x2, y2).
        out_size (int): Output size for the cropped region.
    
    Returns:
        np.ndarray: Cropped frame padded to out_size x out_size.
    """
    fh, fw, _ = frame.shape
    x1, y1, x2, y2 = bbox

    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(fw, x2)
    sy2 = min(fh, y2)

    crop = frame[sy1:sy2, sx1:sx2]

    out = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    ox = sx1 - x1
    oy = sy1 - y1

    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def crop_frame(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Crop a frame using a bounding box and resize to CROP_SIZE.
    
    Args:
        frame (np.ndarray): Input frame to crop.
        bbox (Optional[Tuple[int, int, int, int]]): Bounding box as (x1, y1, x2, y2) or None.
    
    Returns:
        np.ndarray: Cropped and resized frame of size CROP_SIZE x CROP_SIZE x 3.
    """

    if bbox is None:
        return np.zeros((config.CROP_SIZE, config.CROP_SIZE, 3), dtype=np.uint8)

    w, h = bbox_size(bbox)
    crop_size = config.CROP_SIZE

    # Step 0: Expand bbox by safety margin
    x1, y1, x2, y2 = bbox
    side = max(w, h)
    margin = int(side * config.SAFETY_MARGIN)
    # Expand coordinates outward, clamp to frame size
    fh, fw = frame.shape[0], frame.shape[1]
    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(fw, x2 + margin)
    y2m = min(fh, y2 + margin)
    bbox_margin = (x1m, y1m, x2m, y2m)

    # Step 1: Make bbox square if not already
    w, h = bbox_size(bbox_margin)
    if w != h:
        side = max(w, h)
        cx = (bbox_margin[0] + bbox_margin[2]) // 2
        cy = (bbox_margin[1] + bbox_margin[3]) // 2
        half = side // 2
        square_bbox = (
            cx - half,
            cy - half,
            cx - half + side,
            cy - half + side,
        )
    else:
        square_bbox = bbox_margin
        side = w  # == h

    # Step 2: If square bbox is smaller than crop_size, expand symmetrically
    if side < crop_size:
        cx = (square_bbox[0] + square_bbox[2]) // 2
        cy = (square_bbox[1] + square_bbox[3]) // 2
        half = crop_size // 2
        expanded_bbox = (
            cx - half,
            cy - half,
            cx - half + crop_size,
            cy - half + crop_size,
        )
        crop = crop_with_clamp(frame, expanded_bbox, crop_size)
        return crop
    else:
        # Step 3: Crop and resize to crop_size
        crop = crop_with_clamp(frame, square_bbox, side)
        return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def write_cropped(
    video_path: str,
    out_folder: str,
    bboxes: Dict[int, Optional[Tuple[int, int, int, int]]],
    obj_id: int,
) -> str:
    """Write a cropped video based on bounding boxes for a specific object.
    
    Args:
        video_path (str): Path to the input video file.
        out_folder (str): Output folder where the cropped video will be saved.
        bboxes (Dict[int, Optional[Tuple[int, int, int, int]]]): Frame-to-bbox mapping.
        obj_id (int): Object ID for naming the output file.
    
    Returns:
        str: Path to the output cropped video file.
    """

    os.makedirs(out_folder, exist_ok=True)
    logger.info("Writing cropped video for object %s", obj_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Find first frame with a bbox for naming 
    first_frame_idx = next((idx for idx, bb in bboxes.items() if bb is not None), None)
    if first_frame_idx is not None:
        first_bbox = bboxes[first_frame_idx]
        # Format: x1-y1-x2-y2
        bbox_str = "_".join(map(str, first_bbox))
    else:
        bbox_str = "no_bbox"

    # Create filename with bbox
    name = os.path.basename(video_path).replace(".mp4", f"_start{bbox_str}.mp4")

    out_path = os.path.join(out_folder, name)

    out = cv2.VideoWriter(
        os.path.join(out_folder, name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (config.CROP_SIZE, config.CROP_SIZE),
    )

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(crop_frame(frame, bboxes.get(i)))
        i += 1

    cap.release()
    out.release()
    return out_path
