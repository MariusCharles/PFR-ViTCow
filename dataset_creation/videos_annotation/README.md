## Tool to manually annotate Cow behaviour
Annotate manually the cow's behaviour. 

### 1-Requirements 
Download the following packages before launching the tool.
```
conda create --name video_annotator python=3.12
conda activate video_annotator
pip install flask
pip install ffmpeg-python
```

```
mkdir static/videos/original
mkdir static/videos/h264
```
The videos should be dropped into static/original/

### 2-Change the codec format
Modify the codec of the video to be readable with a webserver.

```
bash convert_codec.sh
```
If the videos are located in static/original/ , everything works correctly and converted videos are located into h264/ and directly recognized by flask.

### 2-Launching 
Run the following command
```
python app.py 
```
Open the html adress given by flask (format similar to : http://127.0.0.1:5000 ) into your web browser (or ctrl+right click)
