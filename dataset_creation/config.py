from pathlib import Path

# get project root (current directory of this config file)
PROJECT_ROOT = Path(__file__).resolve().parent

# SFTP PARAMS 
SFTP_USER = "sftpiodaa"
SFTP_HOST = "88.189.55.27"
SFTP_PORT = 22222

REMOTE_DIR = "/PACECOWVID"
UPLOAD_DIR = "/PACECOWVID/ViTCow_upload"
FARM_NAMES = {
            "BUISSON": REMOTE_DIR + "/BUISSON/20241016 - BUISSON/CUT",
            "COPTIERE": REMOTE_DIR + "/COPTIERE",
            "CORDEMAIS": REMOTE_DIR + "/CORDEMAIS",
            "CYPRES": REMOTE_DIR + "/CYPRES",
            "SAULAIE": REMOTE_DIR + "/SAULAIE/20250327 - Saulaie - Regis Bedouet"
            }

# FOLDER MANAGEMENT 
CLIP_FOLDER = PROJECT_ROOT / "clips"
CROP_FOLDER = PROJECT_ROOT / "crops"
LOCAL_TMP_DIR = PROJECT_ROOT / "data"   # for sftp downloads

# EXTRACTION / CROP PARAMS
NUM_FRAMES_PER_CLIP = 20
FRAME_STEP = 4
NUM_CLIP = 10

CROP_SIZE = 224
PROMPT_CLASS = "cow"

# Safety margin for bounding box expansion (fraction of max side)
SAFETY_MARGIN = 0.1  # 10% margin

START = 9
END = 17

# creates folders automatically 
for d in [CLIP_FOLDER, CROP_FOLDER, LOCAL_TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)