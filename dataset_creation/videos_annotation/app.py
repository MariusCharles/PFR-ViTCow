from flask import Flask, render_template, request, redirect, url_for
import csv
import os

app = Flask(__name__)

VIDEO_FOLDER = os.path.join("static", "videos/h264")
CSV_FILE = "annotations.csv"

def get_next_video():
    videos = sorted(os.listdir(VIDEO_FOLDER))
    
    # Read annotated videos
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline="") as f:
            annotated = [row[0] for row in csv.reader(f)]
    else:
        annotated = []

    # Pick first video that is not annotated yet
    for v in videos:
        if v not in annotated:
            return v
    return None  # all videos annotated

@app.route("/", methods=["GET"])
def index():
    video_name = get_next_video()
    if video_name is None:
        return "<h2>All videos have been annotated!</h2>"

    # List of behavior names
    behaviors = [
        "Eating",
        "Drinking",
        "Sleeping",
        "Grooming",
        "Playing",
        "Running",
        "Hiding",
        "Exploring"
    ]

    return render_template("index.html", video_name=video_name, behaviors=behaviors)


@app.route("/annotate", methods=["POST"])
def annotate():
    video_name = request.form["video_name"]
    behavior = request.form["behavior"]

    # Append annotation
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([video_name, behavior])

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
