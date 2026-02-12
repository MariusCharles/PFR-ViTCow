from pathlib import Path

# Get project root (current directory of this config file)
PROJECT_ROOT = Path(__file__).resolve().parent

# SFTP parameters
SFTP_USER = "sftpiodaa"
SFTP_HOST = "88.189.55.27"
SFTP_PORT = 22222

UPLOAD_DIR = "/PACECOWVID/ViTCow_upload"
PRETRAIN_DIR= "pretraining_dataset"

#Name of the folder (farm) used for test
TEST_FOLDER="CORDEMAIS" 