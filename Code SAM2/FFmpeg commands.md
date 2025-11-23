
# 1. Extract  frames from a video in jpg (very lossy)

## 1.1. Extract every frame

```shell
ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
```

## 1.2. Extract every other frame

```bash
ffmpeg -i <your_video>.mp4 -vf "select='not(mod(n,2))'" -q:v 2 -vsync 0 -start_number 0 <output_dir>/'%05d.jpg'
```

## 1.3. Extract every 8th frame

Our video surveillance camera system is 12fps, or 8 fps

```bash
ffmpeg -i <your_video>.mp4 -vf "select=not(mod(n\,8))" -vsync vfr ./img%03d.jpg
```

## 1.4. Use extracted frame to make a new video

```bash
ffmpeg -framerate 1 -i img_%03d.jpg -c:v libx264 -crf 1 -vf scale=2560:1440 -pix_fmt yuv420p -vb 100M <name_out>.mp4
```

# 2. Extract frames from a video in png (better quality)

```bash
ffmpeg -hide_banner -loglevel error -stats -i "input.mp4" -map 0:v:0 -vsync 0 -start_number 0 -vf "setsar=1" "frames/frame_%06d.png"
```

# 3. Check the framerate of a video

```shell
 ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nokey=1:noprint_wrappers=1 <name_video>.mp4 | awk -F/ '{printf "%.3f\n", $1/$2}'
  ```
  
# 4. To change video codec from 265+ to 264

```bash
ffmpeg -err_detect ignore_err -i <video>.mp4 -c copy "<video>_fixed.mp4"
```

# 5. To cut a video

from : <https://shotstack.io/learn/use-ffmpeg-to-trim-video/>

```bash
ffmpeg -i <your_video>.mp4 -ss 00:00:00 -t 00:05:00 -c:v copy -c:a copy <your_video>_5m.mp4
```

# 6. Changing 1440p to 1080p videos

```bash
ffmpeg -i 20240227150313_D01_fixed.mp4 -vf scale=-1:1080 -c:v libx264 -crf 18 -preset veryslow -pix_fmt yuv420p -c:a copy 20240227150313_D01_fixed_1080.mp4
```

Command partly from this issue : <https://superuser.com/questions/714804/converting-video-from-1080p-to-720p-with-smallest-quality-loss-using-ffmpeg>
It is possible to make it faster with the -preset : "You control the tradeoff between video encoding speed and compression efficiency with the `-preset` options. Those are **ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow**. Default is **medium**. The **veryslow** option offers the best compression efficiency (resulting in a smaller file size for the same quality) but it is very slow – as the name says."

1 video (1GB) results in a video of size .... in .... minutes
10 min 1440p file into 1080 with veryfast preset : 1min05 == 10x the speed of the video
10 min 1440p file into 1080 with fast preset : 2min30 == 4x the speed of the video
10 min 1440p file into 1080 with veryslow preset : 15 min == 1.5 the speed of the video

# 7. HEVC to H264

```shell
ffmpeg -i <video> -map 0 -c:v libx265 -crf 20 -vf format=yuv420p -c:a copy output.mkv
```

```ad-Note_blue

- crf is for “quality dial”**.
It sets the target _perceptual quality_ of the output video, and the encoder decides how many bits to spend to maintain that quality.
**Scale:**
    - `0` → lossless (huge files)
    - `18–23` → common high-quality range
    - `51` → worst quality / smallest file

- preset : controls encoding speed vs efficiency (not quality directly).
 Slower presets spend more CPU time **compressing more efficiently**, which means:
    - Smaller files at the _same CRF_.
    - But encoding takes longer.
Faster presets use less CPU time, but files get bigger for the same CRF setting.
**Scale:**
 ultrafast, superfast, veryfast, faster, fast, medium (default), slow, slower, veryslow, placebo
```
