import cv2
# import torch
from ultralytics import YOLO
from blur_face import blur_face
import pathlib, glob
import os
PATH_VIDEOS = "D:/Dataset"
# device = "0" if torch.cuda.is_available() else "cpu"
label_videos = glob.glob(PATH_VIDEOS + "/*/*.mp4")
def proccess_videos(videos_url, name_cam):
    count = 0
    counter = 0
    flag1, flag2 = False, False
    model = YOLO('model/yolov8n.pt')
    try:
        video_capture = cv2.VideoCapture(videos_url)
    except Exception:
        print("Video is not accessed")
        raise
    while True:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
        results = model.track(frame, classes=0, conf=0.5)
        for i in results[0]:
            boxes = i.boxes.numpy()
            x1, y1, x2, y2 = boxes.xyxy[0] if len(boxes) else [0,0,0,0]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            x1, y1, x2, y2 = max(x1,0), max(y1, 0), max(x2, 0), max(y2, 0)
            if x1 and x2 and y1 and y2:
                region = frame[y1:y2, x1:x2]
                blur_photo = blur_face(region)
                if flag1 == False and not os.path.isdir("blur-face/{0}".format(name_cam)):
                    os.mkdir("blur-face/{0}".format(name_cam))
                    flag1 = True

                if flag2 == False and not os.path.isdir("detected_from_video_screen/{0}".format(name_cam)):
                    os.mkdir("detected_from_video_screen/{0}".format(name_cam))
                    flag2 = True

                cv2.imwrite("detected_from_video_screen/{0}/id{1}/photo{2}.jpg".format(name_cam, count, counter), region)
                cv2.imwrite("blur-face/{0}/id{1}/yolo_output{2}.jpg".format(name_cam, count, counter), blur_photo)
                counter += 1
        count += 1

for i in label_videos:
    path = pathlib.Path(i).parts
    proccess_videos(i, path[2])