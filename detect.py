import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
video_camera_capture = cv2.VideoCapture("3.mp4")
counter = 0
while True:
    # Capturing a picture from a video
    
    while video_camera_capture.isOpened():
        ret, frame = video_camera_capture.read()
        if not ret:
            break
        
        result = model.track(source=frame, conf=0.5)
        boxes = result[0].boxes.numpy() # Boxes object for bbox outputs 
        x1, y1, x2, y2 = boxes.xyxy[0] if len(boxes) else [0,0,0,0]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        if x1 and x2 and y1 and y2:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)   
            # вырезаем и сохраняем  найденные объекты
            # в папку detected_from_video_screen. Точка перед ней указывает на текущий каталог
            crop_img = frame[y1:y2, x1:x2]
            cv2.imwrite("detected_from_video_screen/photo_count_{0}.jpg".format(counter), crop_img)
            counter = counter+1

        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_camera_capture.release() 
    cv2.destroyAllWindows()
    break
