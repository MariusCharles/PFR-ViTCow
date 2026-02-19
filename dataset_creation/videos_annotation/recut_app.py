from flask import Flask, render_template, request, redirect, url_for
import csv
import os
import shutil
import subprocess

app = Flask(__name__)

CSV_FILE = "annotations.csv"
SOURCE_VIDEO_FOLDER = os.path.join("static", "videos", "h264")
WORK_COPY_FOLDER = os.path.join("static", "videos", "recut_candidates")
OUTPUT_FOLDER = os.path.join("static", "videos", "recut_final")
FINAL_MERGED_FOLDER = os.path.join("static", "videos", "recut_final_merged")
SEGMENTS_CSV = "recut_segments.csv"

TARGET_BEHAVIORS = [
    "Standing_up",
    "Lying_down",
    "Walking",
    "Scratching-standing",
    "Scratching-lying"
]


def read_annotations():
    rows = []
    if not os.path.exists(CSV_FILE):
        return rows

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            video_name = row[0].strip()
            label = row[1].strip()
            if not video_name:
                continue
            rows.append((video_name, label))
    return rows


def build_candidates(target_behaviors):
    annotations = read_annotations()
    filtered = []
    for video_name, label in annotations:
        if label in target_behaviors:
            filtered.append((video_name, label))

    video_to_labels = {}
    for video_name, label in filtered:
        video_to_labels.setdefault(video_name, set()).add(label)

    candidates = []
    for video_name in sorted(video_to_labels.keys()):
        labels = sorted(video_to_labels[video_name])
        candidates.append({"video_name": video_name, "labels": labels, "label_text": " / ".join(labels)})
    return candidates


def ensure_folders():
    os.makedirs(WORK_COPY_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FINAL_MERGED_FOLDER, exist_ok=True)


def sync_candidate_copies(candidates):
    ensure_folders()
    copied = 0
    missing = 0

    for item in candidates:
        video_name = item["video_name"]
        src = os.path.join(SOURCE_VIDEO_FOLDER, video_name)
        dst = os.path.join(WORK_COPY_FOLDER, video_name)

        if not os.path.exists(src):
            missing += 1
            continue

        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    return copied, missing


def load_segment_map():
    segment_map = {}
    if not os.path.exists(SEGMENTS_CSV):
        return segment_map

    with open(SEGMENTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = (row.get("video_name") or "").strip()
            if not video_name:
                continue
            segment_map[video_name] = {
                "label": (row.get("label") or "").strip(),
                "start": (row.get("start_sec") or "").strip(),
                "end": (row.get("end_sec") or "").strip(),
                "output_file": (row.get("output_file") or "").strip(),
            }
    return segment_map


def save_segment_map(segment_map):
    fieldnames = ["video_name", "label", "start_sec", "end_sec", "output_file"]
    with open(SEGMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video_name in sorted(segment_map.keys()):
            item = segment_map[video_name]
            writer.writerow(
                {
                    "video_name": video_name,
                    "label": item.get("label", ""),
                    "start_sec": item.get("start", ""),
                    "end_sec": item.get("end", ""),
                    "output_file": item.get("output_file", ""),
                }
            )


def recut_keep_single_segment(video_name, start_sec, end_sec):
    input_path = os.path.join(WORK_COPY_FOLDER, video_name)
    output_path = os.path.join(OUTPUT_FOLDER, video_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec}",
        "-to",
        f"{end_sec}",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_selected_video(candidates, requested_video):
    if not candidates:
        return None

    video_names = [c["video_name"] for c in candidates]
    if requested_video in video_names:
        for c in candidates:
            if c["video_name"] == requested_video:
                return c
    return candidates[0]


def split_pending_candidates(candidates, segment_map):
    pending = []
    done = []
    for item in candidates:
        if item["video_name"] in segment_map:
            done.append(item)
        else:
            pending.append(item)
    return pending, done


def get_next_video_name(pending_candidates, current_video_name):
    if not pending_candidates:
        return ""

    names = [item["video_name"] for item in pending_candidates]
    if current_video_name in names:
        current_index = names.index(current_video_name)
        if current_index + 1 < len(names):
            return names[current_index + 1]
    return names[0]


def list_source_videos():
    if not os.path.isdir(SOURCE_VIDEO_FOLDER):
        return []
    return sorted(
        [
            file_name
            for file_name in os.listdir(SOURCE_VIDEO_FOLDER)
            if file_name.lower().endswith(".mp4")
        ]
    )


def build_final_merged_folder():
    ensure_folders()
    segment_map = load_segment_map()
    source_videos = list_source_videos()

    for file_name in os.listdir(FINAL_MERGED_FOLDER):
        if file_name.lower().endswith(".mp4") or file_name.lower().endswith(".csv"):
            os.remove(os.path.join(FINAL_MERGED_FOLDER, file_name))

    edited_count = 0
    original_count = 0
    missing_edited_count = 0
    manifest_rows = [["video_name", "status", "label", "start_sec", "end_sec", "source_file"]]

    for video_name in source_videos:
        dst = os.path.join(FINAL_MERGED_FOLDER, video_name)
        edited_src = os.path.join(OUTPUT_FOLDER, video_name)
        original_src = os.path.join(SOURCE_VIDEO_FOLDER, video_name)
        seg = segment_map.get(video_name)

        if seg and os.path.exists(edited_src):
            shutil.copy2(edited_src, dst)
            edited_count += 1
            manifest_rows.append(
                [
                    video_name,
                    "edited",
                    seg.get("label", ""),
                    seg.get("start", ""),
                    seg.get("end", ""),
                    "recut_final",
                ]
            )
        else:
            shutil.copy2(original_src, dst)
            original_count += 1
            if seg and not os.path.exists(edited_src):
                missing_edited_count += 1
                status = "original_missing_edited_output"
            else:
                status = "original"
            manifest_rows.append(
                [
                    video_name,
                    status,
                    (seg or {}).get("label", ""),
                    (seg or {}).get("start", ""),
                    (seg or {}).get("end", ""),
                    "h264",
                ]
            )

    manifest_path = os.path.join(FINAL_MERGED_FOLDER, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(manifest_rows)

    return {
        "total": len(source_videos),
        "edited": edited_count,
        "original": original_count,
        "missing_edited": missing_edited_count,
        "folder": FINAL_MERGED_FOLDER,
    }


@app.route("/", methods=["GET"])
def index():
    all_candidates = build_candidates(TARGET_BEHAVIORS)
    copied_count, missing_count = sync_candidate_copies(all_candidates)
    segment_map = load_segment_map()

    pending_candidates, done_candidates = split_pending_candidates(all_candidates, segment_map)

    requested_video = request.args.get("video_name", "")
    selected_item = get_selected_video(pending_candidates, requested_video)

    selected_segment = None
    if selected_item:
        selected_segment = segment_map.get(selected_item["video_name"])

    initial_start = "0.00"
    initial_end = "0.00"
    if selected_segment:
        initial_start = selected_segment.get("start", "0.00") or "0.00"
        initial_end = selected_segment.get("end", "0.00") or "0.00"

    return render_template(
        "recut_single_segment.html",
        target_behaviors=TARGET_BEHAVIORS,
        candidates=pending_candidates,
        pending_count=len(pending_candidates),
        done_count=len(done_candidates),
        total_count=len(all_candidates),
        selected_item=selected_item,
        selected_segment=selected_segment,
        initial_start=initial_start,
        initial_end=initial_end,
        copied_count=copied_count,
        missing_count=missing_count,
        message=request.args.get("message", ""),
        error=request.args.get("error", ""),
    )


@app.route("/save_segment", methods=["POST"])
def save_segment():
    video_name = request.form.get("video_name", "").strip()
    label_text = request.form.get("label_text", "").strip()
    start_raw = request.form.get("start", "").strip()
    end_raw = request.form.get("end", "").strip()

    if not video_name:
        return redirect(url_for("index", error="Aucune vidéo sélectionnée"))

    try:
        start_sec = float(start_raw)
        end_sec = float(end_raw)
    except ValueError:
        return redirect(url_for("index", video_name=video_name, error="Début/fin invalides"))

    if start_sec < 0 or end_sec <= start_sec:
        return redirect(url_for("index", video_name=video_name, error="Il faut 0 <= début < fin"))

    input_path = os.path.join(WORK_COPY_FOLDER, video_name)
    if not os.path.exists(input_path):
        return redirect(url_for("index", video_name=video_name, error="Vidéo copie introuvable"))

    try:
        output_path = recut_keep_single_segment(video_name, start_sec, end_sec)
    except subprocess.CalledProcessError:
        return redirect(url_for("index", video_name=video_name, error="Erreur ffmpeg pendant le découpage"))

    segment_map = load_segment_map()
    segment_map[video_name] = {
        "label": label_text,
        "start": f"{start_sec:.2f}",
        "end": f"{end_sec:.2f}",
        "output_file": os.path.basename(output_path),
    }
    save_segment_map(segment_map)

    all_candidates = build_candidates(TARGET_BEHAVIORS)
    pending_candidates, _ = split_pending_candidates(all_candidates, segment_map)

    if len(all_candidates) > 0 and len(pending_candidates) == 0:
        result = build_final_merged_folder()
        msg = (
            f"Découpage enregistré. Toutes les vidéos sont traitées. "
            f"Build automatique terminé: {result['total']} vidéos, {result['edited']} redécoupées, "
            f"{result['original']} originales."
        )
        if result["missing_edited"]:
            msg += f" ({result['missing_edited']} sortie(s) manquante(s), original gardé)"
        return redirect(url_for("index", message=msg))

    next_video_name = get_next_video_name(pending_candidates, video_name)
    return redirect(url_for("index", video_name=next_video_name, message="Découpage enregistré, vidéo suivante"))


@app.route("/build_final", methods=["POST"])
def build_final():
    result = build_final_merged_folder()
    msg = (
        f"Final build ok: {result['total']} vidéos, {result['edited']} redécoupées, "
        f"{result['original']} originales. Dossier: {result['folder']}"
    )
    if result["missing_edited"]:
        msg += f" ({result['missing_edited']} sortie(s) redécoupée(s) manquante(s), original gardé)"

    return redirect(url_for("index", message=msg))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
