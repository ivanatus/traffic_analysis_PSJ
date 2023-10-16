# traffic_analysis_PSJ
Analysis of traffic participants in videos of different parts of a city using YOLOv8 pre-trained model for class Programiranje: skriptni jezici.
Base project is https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.

## Steps to run the code
- Clone the repository
```
git clone https://github.com/ivanatus/traffic_analysis_PSJ
```
- Change directory
```
cd traffic_analysis_PSJ
```
- Install the dependecies
```
pip install -e '.[dev]'
```

- Go into next directory
```
cd ultralytics/yolo/v8/detect
```
- Downloading the DeepSORT Files From The Google Drive 
```
https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing
```
- After downloading the DeepSORT Zip file from the drive, unzip it go into the subfolders and place the deep_sort_pytorch folder into the yolo/v8/detect folder

- Downloading a Sample Video from the Google Drive
```
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```

- Run the code with mentioned command below.

- For yolov8 object detection + Tracking
```
python predict.py model=yolov8l.pt source="test3.mp4" show=True
```
- If you want to run this on your own video, place the video in the folder traffic_analysis_PSJ/ultralytics/yolo/v8/detect (.mp4 format)

```
python predict.py model=yolov8l.pt source="your_video_title.mp4" show=True
```
