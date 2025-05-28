# YOLOv8 Detection Demo
This script allows you to run live object detection using a webcam or a video file with a trained YOLOv8 model. The detections will be visualized in real-time and can optionally be saved as a video.

## Requirements
- Python 3.7+

- Ultralytics YOLO

- OpenCV (cv2)

## Install requirements if needed:

!pip install ultralytics opencv-python
## Trained Model
Ensure you have a trained YOLOv8 model (e.g., yolov8_voc_trained.pt). You can train one on your custom dataset or download a pre-trained model.

## How to Run the Demo
You can run the demo with either webcam or video file as the source.

▶ Using Webcam
python your_script.py --demo --source webcam
Opens the default webcam and performs real-time detection.

Press q to exit.

▶ Using a Video File
python your_script.py --demo --source /path/to/video.mp4
Performs detection on a video file.

## Save Annotated Output
To save the annotated video:
python your_script.py --demo --source /path/to/video.mp4 --save --output result.mp4
The output will be saved to result.mp4 (or the file you specify).

