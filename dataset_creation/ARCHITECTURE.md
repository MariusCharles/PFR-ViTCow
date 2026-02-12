# PFR-ViTCow Pipeline Architecture & Data Flow

## COMPLETE PIPELINE SCHEMA - All Function Calls & Sources

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    MAIN PIPELINE (main.py)                                     │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│  START                                                                                         │
│    │                                                                                           │
│    ├─ list_sftp_videos()                [pipeline/cloud.py]                                    │
│    │  └─ Returns: list[str] → videos_path                                                      │
│    │                                                                                           │
│    └─ FOR EACH video_path in videos_path:                                                      │
│       │                                                                                        │
│       ├─ is_daytime_video(video_path)            [video/clipper.py]                            │
│       │  │                                                                                     │
│       │  ├─ FALSE → continue (skip)                                                            │
│       │  └─ TRUE ↓                                                                             │
│       │                                                                                        │
│       ├─ download_sftp_video(video_path)         [pipeline/cloud.py]                           │
│       │  └─ Returns: local_path                                                                │
│       │                                                                                        │
│       ├─ extract_clips(                          [video/clipper.py]                            │
│       │    local_path,                                                                         │
│       │    config.CLIP_FOLDER,                   [config.py]                                   │
│       │    config.NUM_FRAMES_PER_CLIP,           [config.py]                                   │
│       │    config.FRAME_STEP,                    [config.py]                                   │
│       │    config.NUM_CLIP                       [config.py]                                   │
│       │  )                                                                                     │
│       │                                                                                        │ 
│       ├─ remove_video(local_path)                [pipeline/cloud.py]                           │
│       │                                                                                        │
│       └─ FOR EACH clip in config.CLIP_FOLDER:                                                  │
│          │                                                                                     │
│          ├─ subprocess.run(                      [stdlib]                                      │
│          │    ["python", "process_clip.py", clip_path],                                        │
│          │    capture_output=True,                                                             │
│          │    text=True                                                                        │
│          │  )                                                                                  │
│          │  │                                                                                  │
│          │  ├─ SUBPROCESS: process_clip.py ↓                                                   │
│          │  │  ├─ SAMSession()                   [sam/sam_session.py]                          │
│          │  │  ├─ run_extraction()               [pipeline/extractor.py]                       │
│          │  │  └─ Print JSON to stdout                                                         │
│          │  │                                                                                  │
│          │  └─ Returns: CompletedProcess                                                       │
│          │                                                                                     │
│          ├─ TRY:                                                                               │
│          │  ├─ json.loads(result.stdout[-1])     [stdlib]                                      │
│          │  │  └─ Parse: out_all_paths = [str, ...]                                            │
│          │  │                                                                                  │
│          │  └─ FOR EACH out_path in out_all_paths:                                             │
│          │     ├─ upload_video(out_path)         [pipeline/cloud.py]                           │
│          │     └─ remove_video(out_path)         [pipeline/cloud.py]                           │
│          │                                                                                     │
│          ├─ EXCEPT CalledProcessError:                                                         │
│          │  └─ logging.error(...)                [stdlib]                                      │
│          │     └─ continue (next clip)                                                         │
│          │                                                                                     │
│          └─ FINALLY:                                                                           │
│             └─ remove_video(clip_path)           [pipeline/cloud.py]                           │
│                                                                                                │
│  END                                                                                           │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Function Sources Summary

| Function | Source Module | Called From | Line |
|----------|---------------|-------------|------|
| `list_sftp_videos()` | `pipeline/cloud.py` | `main.py` | 14 |
| `is_daytime_video()` | `video/clipper.py` | `main.py` | 18 |
| `download_sftp_video()` | `pipeline/cloud.py` | `main.py` | 21 |
| `extract_clips()` | `video/clipper.py` | `main.py` | 24-28 |
| `remove_video()` | `pipeline/cloud.py` | `main.py` | 31, 51, 65 |
| `subprocess.run()` | Python stdlib | `main.py` | 42-47 |
| `json.loads()` | Python stdlib | `main.py` | 52 |
| `upload_video()` | `pipeline/cloud.py` | `main.py` | 50 |
| `SAMSession()` | `sam/sam_session.py` | `process_clip.py` | 17 |
| `run_extraction()` | `pipeline/extractor.py` | `process_clip.py` | 19 |

---

## DETAILED SUBPROCESS SCHEMAS

### Subprocess 1: `process_clip.py` - SAM3 Clip Processing

**Execution**: `python process_clip.py {clip_path}`  
**Called from**: [main.py](main.py#L42) via `subprocess.run()`  
**Return value**: Stdout with JSON list of output paths

#### Detailed Processing Flow

```
process_clip.py
│
├─ Input: sys.argv[1] = clip_path (from main.py)
│
├─ Import configuration
│  ├─ config.CROP_FOLDER     [config.py]
│  ├─ config.PROMPT_CLASS    [config.py]
│  └─ config.NUM_FRAMES_PER_CLIP, etc.
│
├─ sam = SAMSession()         [sam/sam_session.py]
│  └─ Initializes SAM3 model with GPU resources
│
├─ run_extraction(sam, video_path, config.CROP_FOLDER, config.PROMPT_CLASS)
│  [pipeline/extractor.py]
│  │
│  ├─ sid = sam.start(video_path)
│  │  └─ Initialize SAM3 on the video clip
│  │
│  ├─ first = sam.add_prompt(sid, prompt)
│  │  └─ Add detection prompt: "cow"
│  │     Returns: {"out_obj_ids": [id1, id2, ...]}
│  │
│  ├─ outputs = sam.propagate(sid)
│  │  └─ Run SAM3 inference across all frames
│  │     Returns: {frame_idx: {obj_id: mask_tensor, ...}, ...}
│  │
│  ├─ outputs = prepare_masks_for_visualization(outputs)
│  │  [sam3.visualization_utils]
│  │  └─ Process masks for visualization/export
│  │
│  ├─ obj_ids = first.get("out_obj_ids", [])
│  │  └─ Extract detected object IDs
│  │
│  ├─ out_all_paths = []
│  │
│  └─ FOR EACH obj_id in obj_ids:
│     │
│     ├─ boxes = {}
│     │
│     ├─ FOR EACH frame_idx, masks in outputs.items():
│     │  │
│     │  ├─ IF obj_id in masks:
│     │  │   └─ boxes[frame_idx] = mask_to_bbox(masks[obj_id])
│     │  │      [video/cropper.py]
│     │  │      └─ Convert mask to bounding box
│     │  │
│     │  └─ ELSE:
│     │     └─ boxes[frame_idx] = None
│     │
│     ├─ IF any(boxes.values()):  [Skip objects with no frames]
│     │  │
│     │  ├─ out_path = write_cropped(video_path, out_folder, boxes, obj_id)
│     │  │  [video/cropper.py]
│     │  │  └─ Crop video segment for detected object
│     │  │     Output: {out_folder}/cropped_{obj_id}.mp4
│     │  │
│     │  └─ out_all_paths.append(out_path)
│     │
│     └─ ELSE: skip object (no detections)
│
├─ print(json.dumps(out_all_paths))
│  └─ OUTPUT TO STDOUT: JSON list
│     Example: ["crop_001.mp4", "crop_002.mp4"]
│
├─ FINALLY:
│  ├─ del sam
│  ├─ torch.cuda.synchronize()
│  ├─ torch.cuda.empty_cache()
│  └─ gc.collect()
│
└─ EXIT
```

#### Key Details

| Property | Value |
|----------|-------|
| **Output Format** | JSON list of file paths (last line of stdout) |
| **Working Directory** | Contains input clip |
| **Environment** | GPU required (CUDA) |
| **Config Used** | `CROP_FOLDER`, `PROMPT_CLASS` |
| **Error Handling** | Errors logged to stderr; exceptions propagate to parent |
| **Resource Cleanup** | GPU/CPU memory freed in finally block |

---

### Subprocess 2: SFTP Operations - `pipeline/cloud.py`

Multiple subprocess calls execute SFTP commands.

#### **2a. List Videos** - `list_sftp_videos()`
**Location**: [pipeline/cloud.py](pipeline/cloud.py#L14-L56)  
**Purpose**: Get all available videos from SFTP server

```
Subprocess Chain:
│
├─ SFTP Command 1: List farm folders
│  Command: sftp -P 22222 sftpiodaa@88.189.55.27 << EOF
│           ls /PACECOWVID/COPTIERE
│           exit
│           EOF
│  Output: stdout lines = ["D01", "D02", "D03", ...]
│
├─ FOR EACH folder (D01, D02, ...):
│  │
│  └─ SFTP Command 2: List videos in folder
│     Command: sftp -P 22222 sftpiodaa@88.189.55.27 << EOF
│              ls /PACECOWVID/COPTIERE/{folder}
│              exit
│              EOF
│     Output: stdout lines = ["{filename}.mp4", ...]
│
└─ Aggregate all video paths
   Returns: list[str] = [video1, video2, ...]
```

#### **2b. Download Video** - `download_sftp_video(remote_path)`
**Location**: [pipeline/cloud.py](pipeline/cloud.py#L58-L65)  
**Purpose**: Download single video from SFTP to local disk

```
Subprocess:
│
├─ Input: remote_path = "/PACECOWVID/COPTIERE/D01/202502051430_D01.mp4"
│
├─ Extract filename: "202502051430_D01.mp4"
│
├─ Construct local_path: "/home/kind_boyd/workdir/data/202502051430_D01.mp4"
│
├─ SFTP Command:
│  echo 'get {remote_path} {local_path}' | 
│  sftp -P 22222 sftpiodaa@88.189.55.27
│
├─ Process: File is downloaded to local disk
│
└─ Returns: local_path = "/home/kind_boyd/workdir/data/202502051430_D01.mp4"
```

#### **2c. Upload Video** - `upload_video(out_path)`
**Location**: [pipeline/cloud.py](pipeline/cloud.py#L73)  [Function definition not shown but used]  
**Purpose**: Upload processed video to SFTP server

```
Subprocess:
│
├─ Input: out_path = "/home/kind_boyd/workdir/crop/crop_001.mp4"
│
├─ Extract filename: "crop_001.mp4"
│
├─ SFTP Command:
│  echo 'put {out_path} /PACECOWVID/ViTCow_upload/{filename}' | 
│  sftp -P 22222 sftpiodaa@88.189.55.27
│
└─ Process: File is uploaded to SFTP server
```

#### **2d. Delete File** - `remove_video(path)`
**Location**: [pipeline/cloud.py](pipeline/cloud.py#L66-L68)  
**Purpose**: Delete local file (safe delete with existence check)

```
Function:
│
├─ Input: path = file to delete
│
├─ Check: os.path.exists(path)
│
├─ IF EXISTS:
│  └─ os.remove(path)
│
└─ IF NOT EXISTS: do nothing
```

#### SFTP Configuration

```
SFTP_HOST     = "88.189.55.27"
SFTP_PORT     = 22222
SFTP_USER     = "sftpiodaa"
REMOTE_DIR    = "/PACECOWVID/COPTIERE"        [Source videos]
UPLOAD_DIR    = "/PACECOWVID/ViTCow_upload"   [Processed videos]
LOCAL_TMP_DIR = "/home/kind_boyd/workdir/data" [Local storage]
```

---

## DATA FLOW SUMMARY

```
SFTP Server
    │
    ├─ list_sftp_videos()  ──────────────────► [Get list of videos]
    │
    ├─ for each video:
    │   │
    │   ├─ download_sftp_video()  ───────────► Local Disk
    │   │
    │   ├─ extract_clips()  ─────────────────► Local Clips
    │   │                                       (temporary)
    │   ├─ for each clip:
    │   │   │
    │   │   └─ subprocess: process_clip.py ──► Cropped Videos
    │   │                                       (local)
    │   └─ upload_video()  ──────────────────► SFTP Server
    │       (upload cropped)
    │
    └─ remove_video()  ──────────────────────► Cleanup
```

---

## Configuration Reference

```
config.py:
├─ CLIP_FOLDER = "/teamspace/studios/this_studio/clip"
├─ NUM_FRAMES_PER_CLIP = 20
├─ FRAME_STEP = 4
├─ NUM_CLIP = 10
├─ CROP_FOLDER = "/teamspace/studios/this_studio/crop"
└─ PROMPT_CLASS = "cow"
```

---
Généré à l'aide du copilot VScode
