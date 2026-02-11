# SAM3 Video Segmentation Pipeline

This repository implements a **complete video segmentation and cropping pipeline** built on top of **SAM3 (Segment Anything Model 3)**.

It allows you to:
- split long videos into short clips
- segment objects from a **text prompt**
- track those objects across the full clip
- generate per-object cropped output videos

This repository acts as an **orchestration layer** around the official **SAM3** repository.

---

##  Features

- Long video clipping into fixed-length clips
- Text-prompt–guided video segmentation (SAM3)
- Multi-frame instance tracking
- Per-frame bounding box extraction
- Video cropping with padding and black background
- Explicit GPU (CUDA) memory management

---

## Running the Pipeline : On `https://mydocker.centralesupelec.fr`

### 1. Environment

We create a venv to isolate the environment. First install python and dependencies : 

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv libgl1 libglib2.0-0 git
```

Then, create the environment and download requirements : 

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements_vm.txt
```

### 2. Clone the Repo 

```bash
git clone https://github.com/MariusCharles/PFR-ViTCow.git
cd PFR-ViTCow/
```


### 3. Access Requirements (Hugging Face)

SAM3 is a **gated model** hosted on Hugging Face.  
Before using this pipeline, you must **request access** and **authenticate locally**.

#### A- Request access to SAM3

You must request access on the official SAM3 Hugging Face page  
(approval required by Meta / Facebook Research).

> ⚠️ Without approval, the model weights cannot be downloaded.

#### B- Authenticate with your Hugging Face account

Once access is granted, log in locally (change "your_token" with the token obtained on huggingface website):

```bash
python - << 'EOF'
from huggingface_hub import login
login(token="your_token")
print("HF login successful")
EOF
```


### 4. Choose `max_num_objects`

You can control the maximum number of objects detected by SAM3 using the `max_num_objects` parameter.  
This helps regulate GPU memory usage and allows you to limit how many cows are kept per clip.

To modify this parameter, we update an internal SAM3 configuration value. ⚠️ Be careful if you cloned multiple SAM3 repos, to indicate the correct path (usually in the .venv folder). ⚠️

To update the maximum number of objects handled by SAM3, run:

```bash
bash update_max_objects.sh <path_to_sam3_repo> <max_num_objects>
```

For example, to set `max_num_objects` to 10 on `centrale mydocker` :

```bash
bash update_max_objects.sh .venv/src/sam3/ 10
```

### 5. Running the Pipeline

The main entry point is:

```bash
python main.py
```


## Running the Pipeline : Locally or on Lightning AI 

### 1. Environment

- **Python 3.12 (required)**
- NVIDIA GPU recommended (CUDA)

If you are running the code locally, create a dedicated Conda environment:

```bash
conda create -n sam3-pipeline python=3.12
conda activate sam3-pipeline
```

If you are running this project on Lightning AI (Studio / CloudSpace), skip this step.

---

### 2. Clone the Repo 

```bash
git clone https://github.com/MariusCharles/PFR-ViTCow.git
cd PFR-ViTCow/
```

---

### 3. Python Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 4. External Dependency: SAM3

SAM3 is **not distributed on PyPI** and must be cloned from the official repository.

#### Official repository

 https://github.com/facebookresearch/sam3

#### Installation

```bash
git clone https://github.com/facebookresearch/sam3.git
pip install -e sam3
```

The `sam3/` directory must be present in the directory or otherwise available in your `PYTHONPATH`.
If you encounter dependency issues at this step, retry the previous step using the alternative requirements file.

---

### 3bis & 4bis. If it fails

Skip this section if everything is working correctly.

If you encounter dependency conflicts (including when running `main.py`), try installing the alternative requirements file:

```bash
pip install -r requirements_v2.txt
```

⚠️ **Important:**  
This will install both the Python dependencies **and** the SAM3 repository as a package.  
In this setup, SAM3 will be installed inside a `src/` directory.

Keep this in mind when modifying the `max_num_objects` parameter, as the file path to SAM3 may differ from the default setup.

---

### 5. Access Requirements (Hugging Face)

SAM3 is a **gated model** hosted on Hugging Face.  
Before using this pipeline, you must **request access** and **authenticate locally**.

#### A- Request access to SAM3

You must request access on the official SAM3 Hugging Face page  
(approval required by Meta / Facebook Research).

> ⚠️ Without approval, the model weights cannot be downloaded.

#### B- Authenticate with your Hugging Face account

Once access is granted, log in locally:

```bash
huggingface-cli login
```

If it fails, try : 

```bash
python - << 'EOF'
from huggingface_hub import login
login(token="your_token")
print("HF login successful")
EOF
```

---

### 6. Choose `max_num_objects`

You can control the maximum number of objects detected by SAM3 using the `max_num_objects` parameter.  
This helps regulate GPU memory usage and allows you to limit how many cows are kept per clip.

To modify this parameter, we update an internal SAM3 configuration value.

To update the maximum number of objects handled by SAM3, run:

```bash
bash update_max_objects.sh <path_to_sam3_repo> <max_num_objects>
```

For example, to set `max_num_objects` to 5, in a LightningAI setup:

```bash
bash update_max_objects.sh /teamspace/studios/this_studio/PFR-ViTCow/sam3 5
```

> This modification will later be moved directly into the configuration file.

---

### 7. Running the Pipeline

The main entry point is:

```bash
python main.py
```

#### Configuration

Pipeline parameters are defined in `config.py`:

```python
INPUT_FOLDER = "/path/to/input/videos"
CLIP_FOLDER = "/path/to/clips"
CROP_FOLDER = "/path/to/crops"

NUM_FRAMES_PER_CLIP = 20
FRAME_STEP = 4

CROP_SIZE = 224

PROMPT_CLASS = "cow"
```

---

##  Repository Architecture

### Directory Structure

```text
repo/
├── main.py                 # Global pipeline orchestration
├── config.py               # Global configuration
├── requirements.txt
│
├── sam/                     # SAM3 wrapper
│   ├── sam_session.py       # SAM3 video session handling
│   └── __init__.py
│
├── video/                   # Video processing utilities
│   ├── clipper.py           # Video clipping
│   ├── cropper.py           # Bounding boxes, cropping, video writing
│   └── __init__.py
│
├── pipeline/
│   ├── extractor.py         # SAM → BBox → Crop pipeline
│   └── __init__.py
│
├── sam3/                    # Official SAM3 repository (git clone)
└── README.md
```

---

##  Architecture Diagram

```text
┌────────────────────┐
│   Input Videos     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  video/clipper     │  Video clipping
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  SAMSession        │  (sam/sam_session.py)
│  - start_session   │
│  - add_prompt      │
│  - propagate       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ pipeline/extractor │
│ - masks → bbox     │
│ - padding          │
│ - crop             │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Cropped Videos    │
└────────────────────┘
```

---


## 📄 License

This repository provides an application-level pipeline.

The model weights and license are defined by the official **SAM3 (Facebook Research)** repository.

---

##  References

- SAM3: https://github.com/facebookresearch/sam3
- Segment Anything: https://github.com/facebookresearch/segm

