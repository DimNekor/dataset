import cv2
from ultralytics import YOLO
from blur_face import blur_face
import pathlib, glob
import os
import torch
PATH_VIDEOS = "D:/Dataset1"
device = "0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
label_videos = glob.glob(PATH_VIDEOS + "/*/*.mp4")

def process_frame(frame):
    pass

if not os.path.isdir("blur-face"):
    os.mkdir("blur-face")
if not os.path.isdir("detected_from_video_screen"):
    os.mkdir("detected_from_video_screen")

print(label_videos)
def proccess_videos(videos_url, name_cam):
    numOfFrames = 0
    counter = 0
    flag1, flag2 = False, False
    model = YOLO('model/yolov8n.pt').to(device)
    try:
        video_capture = cv2.VideoCapture(videos_url)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        print("Video is not accessed")
        raise
    

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if numOfFrames != 0:
            process_frame(frame)
        else:
            
            if not ret:
                print("NO FRAME")
                break
            results = model.track(source=frame, conf=0.5, classes=0)
            track_obj = results[0].boxes.numpy()

            track_id = track_obj.id
            coord = track_obj.xyxy
            x1, y1, x2, y2 = coord[0] if len(coord) else [0,0,0,0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x1 and x2 and y1 and y2:
                region = frame[y1:y2, x1:x2]
                blur_photo = blur_face(region)

                if flag1 == False and not os.path.isdir("blur-face/{0}".format(name_cam)):
                    os.mkdir("blur-face/{0}".format(name_cam))
                    flag1 = True

                if flag2 == False and not os.path.isdir("detected_from_video_screen/{0}".format(name_cam)):
                    os.mkdir("detected_from_video_screen/{0}".format(name_cam))
                    flag2 = True

                if not os.path.isdir('detected_from_video_screen/{0}/id{1}'.format(name_cam, track_id)):
                    os.mkdir('detected_from_video_screen/{0}/id{1}'.format(name_cam, track_id))

                if not os.path.isdir("blur-face/{0}/id{1}".format(name_cam, track_id)):
                    os.mkdir("blur-face/{0}/id{1}".format(name_cam, track_id))

                cv2.imwrite("detected_from_video_screen/{0}/id{1}/photo{2}.jpg".format(name_cam, track_id, counter), region)
                cv2.imwrite("blur-face/{0}/id{1}/yolo_output{2}.jpg".format(name_cam, track_id, counter), blur_photo)
                counter += 1
            # video_capture.release()
        numOfFrames+=1
        numOfFrames%=10
# cv2.destroyAllWindows()

for i in range(len(label_videos)):
    print(label_videos[i])
    path = pathlib.Path(label_videos[i]).parts
    proccess_videos(label_videos[i], path[2])