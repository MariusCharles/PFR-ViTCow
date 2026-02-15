import os
import subprocess

# Dossiers source et destination
SRC_FOLDER = "static/videos/original"
DST_FOLDER = "static/videos/h264"

# Crée le dossier de destination si inexistant
os.makedirs(DST_FOLDER, exist_ok=True)

# Parcours tous les fichiers MP4 dans le dossier source
for filename in os.listdir(SRC_FOLDER):
    if filename.lower().endswith(".mp4"):
        src_path = os.path.join(SRC_FOLDER, filename)
        dst_path = os.path.join(DST_FOLDER, filename)

        print(f"Conversion : {filename} → {dst_path}")

        # Commande ffmpeg
        command = [
            "ffmpeg",
            "-i", src_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            dst_path
        ]

        # Exécute la commande et attend la fin
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Conversion terminée : {filename}")
        else:
            print(f"❌ Erreur pour {filename} :")
            print(result.stderr)
