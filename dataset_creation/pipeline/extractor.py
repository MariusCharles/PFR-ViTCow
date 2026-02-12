import logging
from video.cropper import mask_to_bbox, write_cropped
from sam3.visualization_utils import prepare_masks_for_visualization
import json
import sys

# Logger vers stderr pour ne pas polluer stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_extraction(sam, video_path: str, out_folder: str, prompt: str) -> None:
    """Run SAM-based object extraction and video cropping on a single video.
    
    Args:
        sam: SAM session object for video processing.
        video_path (str): Path to the input video file.
        out_folder (str): Output folder where cropped videos will be saved.
        prompt (str): Text prompt for object detection and segmentation.
    
    Returns:
        None: Outputs JSON list of cropped video paths to stdout.
    """
    logger.info("Starting extraction for %s", video_path)

    sid = sam.start(video_path)
    first = sam.add_prompt(sid, prompt)
    outputs = sam.propagate(sid)

    outputs = prepare_masks_for_visualization(outputs)
    obj_ids = first.get("out_obj_ids", [])

    out_all_paths = []

    for obj_id in obj_ids:
        boxes = {}
        for idx, masks in outputs.items():
            boxes[idx] = mask_to_bbox(masks[obj_id]) if obj_id in masks else None
        if not any(boxes.values()):
            continue

        out_path = write_cropped(video_path, out_folder, boxes, obj_id)
        out_all_paths.append(out_path)

    # Seul print sur stdout, pour le pipeline
    print(json.dumps(out_all_paths))
