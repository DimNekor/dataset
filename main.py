import os
from threading import Thread
import cv2
import re
from ultralytics import YOLO

PATH_VIDEOS = "D:/Dataset"
PATH_DETECTED = "detected_from_video_screen"

label_videos = []
label_frames = []
threads = []
threads2 = []

for dirpath, dirnames, filenames in os.walk(PATH_VIDEOS):
    for filename in filenames:
        string = os.path.join(dirpath, filename)
        result = re.search(r'Kam\d', string)
        label_videos.append([string, result.group(0)])

def process_videos(path_to_file):
    model = YOLO('yolov8n.pt')
    try:
        video_camera_capture = cv2.VideoCapture(path_to_file[0])
    except Exception:
        print("Video is not accessed")
    counter = 0
    while True:
        while video_camera_capture.isOpened():
            ret, frame = video_camera_capture.read()
            if not ret:
                break
            
            result = model.track(source=frame, conf=0.5, classes=0)
            boxes = result[0].boxes.numpy() # Boxes object for bbox outputs 
            x1, y1, x2, y2 = boxes.xyxy[0] if len(boxes) else [0,0,0,0]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            if x1 and x2 and y1 and y2:
                crop_img = frame[y1:y2, x1:x2]
                if not os.path.isdir("detected_from_video_screen/{0}".format(path_to_file[1])):
                    os.mkdir("detected_from_video_screen/{0}".format(path_to_file[1]))
                cv2.imwrite("detected_from_video_screen/{0}/photocount{1}.jpg".format(path_to_file[1], counter), crop_img)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                counter = counter+1

                cv2.imshow("camera", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        video_camera_capture.release() 
        cv2.destroyAllWindows()
        break


for dirpath, dirnames, filenames in os.walk(PATH_DETECTED):
    for dirname in dirnames:
        for i, j, filename in os.walk(PATH_DETECTED + "/" + dirname):
            label_frames.append((filename, dirname))        
def yolo_face_blur(path_to_file):
    counter = 0
    weights_path = "pretrained_models/YOLO_Model.pt"

    model = YOLO('model/yolov8n.pt')
    model = YOLO(weights_path)
    obj = path_to_file
    img_path = [0]
    for i in img_path:
        pic = PATH_DETECTED + "/" + obj[1] + "/" + i
        results = model(pic)
        img = cv2.imread(pic)

        # loop over detections
        for result in results:
            
            boxes = result.boxes

            if boxes:

                for box in boxes:

                    bouding_cordinate = box.xyxy
                    x1, y1, x2, y2 = ( int(bouding_cordinate[0,0].item()), int(bouding_cordinate[0,1].item()), 
                                    int(bouding_cordinate[0,2].item()), int(bouding_cordinate[0,3].item()) )

                    # delete all the negative cordinates
                    x1, y1, x2, y2 = max(x1,0), max(y1, 0), max(x2, 0), max(y2, 0)

                    # select the region & apply Gaussian Blur & put it on the original image
                    region = img[y1:y2, x1:x2]
                    blurred_region = cv2.GaussianBlur(region, (25,25), 100)
                    img[y1:y2, x1:x2] = blurred_region

        if not os.path.isdir("blur-face/{0}".format(obj[1])):
            os.mkdir("blur-face/{0}".format(obj[1]))
        cv2.imwrite("blur-face/{0}/yolo_output{1}.jpg".format(obj[1], counter), img)
        counter+=1

for i in label_videos:
    threads.append(Thread(target=process_videos, args=(i, )))
for i in threads:
    i.start()
for dirpath, dirnames, filenames in os.walk(PATH_DETECTED):
    for dirname in dirnames:
        for i, j, filename in os.walk(PATH_DETECTED + "/" + dirname):
            label_frames.append((filename, dirname)) 

for i in label_frames:
    threads2.append(Thread(target=yolo_face_blur, args=(i, )))
for i in threads2:
    i.start()