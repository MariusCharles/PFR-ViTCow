#### SAM2 tracker was made with another repo : https://github.com/J-ally/sam2_local
#### SAM2 packaged with the transformers library from hugging face : huggingface.co/facebook/sam2.1-hiera-large

from transformers import Sam2VideoModel, Sam2VideoProcessor
import matplotlib.pyplot as plt
import time
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import gc

# Clear GPU cache
torch.cuda.empty_cache()

"""
Script used to track multiple objects in a video using SAM2 model from Hugging Face.
It initializes the tracker with ground truth boxes for the first frame and propagates masks through the video using a 'streaming'
approach (batching frames to fit in GPU memory - can be choosen with the BATCH_SIZE_SAM2_VIDEO argument depending on the VRAM of the GPU).
Outputs a CSV file with bounding box coordinates for each tracked object in each frame.
"""

SAM2_MODEL = "facebook/sam2.1-hiera-large"
BATCH_SIZE_SAM2_VIDEO = 100
VIDEO_PATH = ""
GT_BOXES_PATH = ""  # csv with the ground truth annotations for the first frame
# Can also use the convert_normalized_to_pixels function in the video_predictor_serv.py


###############################################################################
#                                 FUNCTIONS                                   #
###############################################################################


def mask_to_bbox(mask, threshold=0.0) -> list | None:
    """Convert a mask to a bounding box [x1, y1, x2, y2]

    Args:
        mask: Mask tensor/array with logit values
        threshold: Threshold value to binarize the mask (default: 0.0 for logits)
    """
    # Convert to numpy if it's a tensor
    if torch.is_tensor(mask):
        mask = mask.float().cpu().numpy()  # Convert bfloat16 to float32 first

    # Handle the correct number of dimensions
    if mask.ndim == 4:
        binary_mask = mask[0, 0] > threshold  # Shape: (H, W)
    elif mask.ndim == 3:
        binary_mask = mask[0] > threshold
    else:
        binary_mask = mask > threshold

    # Find coordinates of True elements (the mask)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None  # Return None for empty mask

    # Find min/max positions (bounding box corners)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def convert_norm_bbox_to_pixels(normalized_box, img_width=2560, img_height=1440):
    """
        Convert normalized YOLO format [class_id, x_center, y_center, width, height] to
        pixel coordinates [x1, y1, x2, y2]

        Parameters:
        normalized_box (str): Space-separated string with format "class_id x_center y_center width height"
        img_width (int): Width of the image in pixels
        img_height (int): Height of the image in pixels

        Returns:
        list: Pixel coordinates [x1, y1, x2, y2]


    # Example using a yolo annotation format bounding box
    normalized_box_str = "1 0.14062499999999997 0.39129044792991385 0.2189732142857142 0.30758089585982773
        2 0.34402901785714285 0.2625 0.11350446428571427 0.2321428571428571
        3 0.5143500813057096 0.09639695254155617 0.07427276559865092 0.10863346181642963
        4 0.6826801517067003 0.45399634780165754 0.12895069532237624 0.38432364096080884
        5 0.07016434892541097 0.19441288033566656 0.14032869785082194 0.11462994665251108
        6 0.7759125639839426 0.4714145245118699 0.1308550237388152 0.24610198061525504"
    pixel_box = convert_normalized_to_pixels(normalized_box_str)

    """

    all_bounding_boxes = []

    for box in normalized_box.split("\n"):
        values = box.split()
        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        # Convert center coordinates to absolute pixels
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height

        # Convert width/height to absolute pixels
        width_px = width * img_width
        height_px = height * img_height

        # Calculate corner coordinates
        x1 = x_center_px - (width_px / 2)
        y1 = y_center_px - (height_px / 2)
        x2 = x_center_px + (width_px / 2)
        y2 = y_center_px + (height_px / 2)

        all_bounding_boxes.append([x1, y1, x2, y2])

    return all_bounding_boxes


def plot_mask_on_image(
    mask, frame_id, video_url, obj_id=None, alpha=0.5, colormap="jet"
):
    def get_frame(video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None
        return frame

    # get the n frame
    frame_rgb = get_frame(video_url, frame_id)

    binary_mask = (mask > 0).cpu().numpy().astype(np.uint8)

    # Create filename suffix with object ID if provided
    suffix = f"_obj_{obj_id}" if obj_id is not None else ""

    plt.close("all")

    # Create visualization
    # fig1 = plt.figure(figsize=(12, 8))

    # Option 1: Overlay with transparency
    # plt.imshow(frame_rgb)
    # plt.imshow(
    #     binary_mask[0, 0], cmap=colormap, alpha=alpha
    # )  # Use 'jet', 'hot', or 'viridis' colormap
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(
    #     f"frame_{frame_id}{suffix}_mask_overlay.png", dpi=150, bbox_inches="tight"
    # )
    # plt.close(fig1)

    # Option 2: Side by side comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(frame_rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(binary_mask[0, 0], cmap=colormap)
    ax2.set_title("Mask")
    ax2.axis("off")

    ax3.imshow(frame_rgb)
    ax3.imshow(binary_mask[0, 0], cmap=colormap, alpha=alpha)
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"frame_{frame_id}{suffix}_mask_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    overall_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Sam2VideoModel.from_pretrained(SAM2_MODEL).to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(SAM2_MODEL)

    df_gt_annotations = pd.read_csv(GT_BOXES_PATH, header=0, sep=",")

    first_annotations = df_gt_annotations[df_gt_annotations["frame"] == 1]

    first_annotations_array = np.array(
        [
            [row["x1"], row["y1"], row["x2"], row["y2"]]
            for _, row in first_annotations.iterrows()
        ],
        dtype=np.float32,
    )
    print(f"First frame annotations:\n{first_annotations_array}")

    # First frame annotations:
    # [[1852.8197   575.2169  2164.1824   982.4249 ]
    #  [1169.3184   885.4527  1888.0977  1440.     ]
    #  [ 914.0997   552.1058  1132.8586   770.8647 ]
    #  [1052.1261   255.21872 1203.174    476.58188]
    #  [ 130.21364  830.763    710.96643 1432.35   ]
    #  [1526.1038   276.0529  1783.9268   523.4588 ]]

    all_labels = [
        int(x) for x in first_annotations["track_id"].values
    ]  # Convert to native Python ints
    print(f"Tracking {len(all_labels)} instances: {all_labels}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    all_segments = {}
    last_masks = {}  # Store last masks for each object ID

    all_results = []

    for batch_start in range(0, total_frames, BATCH_SIZE_SAM2_VIDEO):
        print(
            f"Processing frames {batch_start} to {batch_start + BATCH_SIZE_SAM2_VIDEO}"
        )

        # Load batch of frames
        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, batch_start)

        original_height, original_width = (
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        )

        batch_frames = []
        for _ in range(BATCH_SIZE_SAM2_VIDEO):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(Image.fromarray(frame_rgb))

        cap.release()

        if not batch_frames:
            break

        # Process this batch
        inference_session = processor.init_video_session(
            video=batch_frames,
            inference_device=device,
            dtype=torch.bfloat16,
        )

        if batch_start == 0:  # Initialize everything for the first batch
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=0,
                obj_ids=all_labels[:],
                input_boxes=[first_annotations_array],
            )

        else:  # Subsequent batches: use only objects that were tracked in previous batch
            last_mask_key = list(last_masks.keys())[-1]

            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=0,
                obj_ids=all_labels[:],
                input_masks=[
                    last_masks[last_mask_key][x]
                    for x in range(len(last_masks[last_mask_key]))
                ],
            )

        # Clear last_masks AFTER using it for initialization
        last_masks.clear()

        # Propagate through batch
        for sam2_video_output in model.propagate_in_video_iterator(
            inference_session, start_frame_idx=0
        ):
            video_res_masks = processor.post_process_masks(
                [sam2_video_output.pred_masks],
                original_sizes=[
                    [inference_session.video_height, inference_session.video_width]
                ],
                binarize=False,
            )[0]

            # Store masks for this frame
            frame_idx = batch_start + sam2_video_output.frame_idx
            all_segments[frame_idx] = video_res_masks

        last_masks[frame_idx] = video_res_masks  # Store last masks for next batch

        # print(last_masks.keys())
        # print(last_masks)
        # print(len(last_masks[99]))
        # print(last_masks[99][0].shape)

        # Clear memory after each batch
        del inference_session, batch_frames
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Tracked {len(all_labels)} objects through {len(all_segments)} frames total")

    frame_to_plot = [0, 50, 100, 150, 200, 250]

    # for frame in frame_to_plot:
    #     if frame in all_segments:
    #         # Plot each instance separately or combined
    #         for idx, obj_id in enumerate(all_labels):
    #             if idx < len(all_segments[frame]):
    #                 plot_mask_on_image(
    #                     all_segments[frame][idx : idx + 1],
    #                     frame,
    #                     VIDEO_PATH,
    #                     obj_id=obj_id,
    #                 )

    # Convert masks to bounding boxes for all instances
    bboxes = []
    for frame_idx, masks in all_segments.items():
        # print(f"Processing frame {frame_idx} with {len(masks)} masks")
        # print(f"first mask shape: {masks[0].shape}")
        for idx, obj_id in enumerate(all_labels):
            if idx < len(masks):
                bbox = mask_to_bbox(masks[idx : idx + 1])
                if bbox is not None:
                    segment_bboxes = {
                        "frame": frame_idx,
                        "track_id": obj_id,
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "confidence": 1.0,
                        "class": 19,  # the class ID for cow in COCO dataset
                        "class_name": "cow",
                    }
                    bboxes.append(segment_bboxes)

    df_SAM2_results = pd.DataFrame(bboxes)
    print(df_SAM2_results.head(20))  # Show results for multiple instances

    df_SAM2_results.to_csv(
        "data/SAM2_video_tracking_results.csv",
        index=False,
    )

    overall_end = time.time()
    print(f"Overall processing time: {overall_end - overall_start:.2f} seconds  ")
