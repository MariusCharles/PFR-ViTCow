import logging
import cv2

logger = logging.getLogger(__name__)

def extract_batches(video_path: str, batch_size: int):
    """Extract fixed-size frame batches from a video.

    Frames are read sequentially from the input video and grouped into
    batches of length `batch_size`. Batches are yielded as soon as they
    are complete, enabling streaming processing. The final batch may
    contain fewer frames.

    Args:
        video_path: Path to the input video file.
        batch_size: Number of frames per batch.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    batch = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if batch:
                    yield batch
                break

            batch.append(frame)
            if len(batch) == batch_size:
                yield batch
                batch = []
    finally:
        cap.release()