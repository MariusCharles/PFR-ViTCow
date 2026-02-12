for f in static/videos/original/*.mp4; do
    filename=$(basename "$f")  # extract just the filename
    ffmpeg -i "$f" \
        -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
        -c:a aac \
        "static/videos/h264/$filename"
done
