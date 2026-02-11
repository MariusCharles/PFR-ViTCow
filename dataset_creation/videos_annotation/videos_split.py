import paramiko
import random
import getpass

SFTP_HOST = "88.189.55.27"
SFTP_PORT = 22222
SFTP_USER = "sftpiodaa"

source_folders = [
    "/PACECOWVID/ViTCow_upload/BUISSON",
    "/PACECOWVID/ViTCow_upload/COPTIERE",
    "/PACECOWVID/ViTCow_upload/CORDEMAIS",
    "/PACECOWVID/ViTCow_upload/CYPRES",
    "/PACECOWVID/ViTCow_upload/SAULAIE",
]

people = ["Marie", "Marius", "Mafalda", "Charlotte", "Damien"]
VIDEOS_PER_PERSON = 100

output_base = "/PACECOWVID/ViTCow_upload/annotation"
log_file_path = "/PACECOWVID/ViTCow_upload/annotation/distributed_clips.txt"

# ---- ASK PASSWORD ----
password = getpass.getpass("SFTP password: ")

# ---- CONNECT ----
transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
transport.connect(username=SFTP_USER, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)

distributed_clips = []

# ---- CREATE BASE OUTPUT FOLDER ----
try:
    sftp.mkdir(output_base)
except IOError:
    pass

# ---- CREATE PERSON FOLDERS ----
for person in people:
    try:
        sftp.mkdir(f"{output_base}/{person}")
    except IOError:
        pass

# ---- DISTRIBUTE VIDEOS ----
for folder in source_folders:
    videos = sftp.listdir(folder)
    random.shuffle(videos)

    required = VIDEOS_PER_PERSON * len(people)
    if len(videos) < required:
        raise RuntimeError(f"Not enough videos in {folder}")

    for i, person in enumerate(people):
        selected = videos[i*VIDEOS_PER_PERSON:(i+1)*VIDEOS_PER_PERSON]

        for video in selected:
            src = f"{folder}/{video}"
            dst = f"{output_base}/{person}/{video}"
            sftp.rename(src, dst)
            distributed_clips.append(video)

# ---- WRITE LOG FILE ----
with sftp.open(log_file_path, "w") as f:
    for clip in distributed_clips:
        f.write(clip + "\n")

# ---- CLEANUP ----
sftp.close()
transport.close()
