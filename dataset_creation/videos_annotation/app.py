from flask import Flask, render_template, request, redirect, url_for
import csv
import os

app = Flask(__name__)

VIDEO_FOLDER = os.path.join("static", "videos/h264")
CSV_FILE = "annotations.csv"

def get_next_video():
    videos = sorted(os.listdir(VIDEO_FOLDER))
    
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline="") as f:
            annotated = [row[0] for row in csv.reader(f)]
    else:
        annotated = []

    for v in videos:
        if v not in annotated:
            return v
    return None


@app.route("/", methods=["GET"])
def index():
    video_name = get_next_video()
    if video_name is None:
        return "<h2>All videos have been annotated!</h2>"

    behaviors = [
        "Standing",
        "Standing_up",
        "Lying",
        "Lying_down",
        "Walking",
        "Feeding",
        "Drinking",
        "Mounting",
        "Grooming-standing",
        "Grooming-lying",
        "Self-grooming-standing",
        "Self-grooming-lying",
        "Scratching-standing",
        "Scratching-lying",
        "Delete"
    ]

    return render_template("index.html", video_name=video_name, behaviors=behaviors)


@app.route("/annotate", methods=["POST"])
def annotate():
    video_name = request.form["video_name"]
    behavior = request.form["behavior"]

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([video_name, behavior])

    return redirect(url_for("index"))


@app.route("/undo", methods=["POST"])
def undo():
    if not os.path.exists(CSV_FILE):
        return redirect(url_for("index"))

    with open(CSV_FILE, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return redirect(url_for("index"))

    last_video = rows[-1][0]
    rows = rows[:-1]

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Important: redirect instead of render_template
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
