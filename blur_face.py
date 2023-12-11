from ultralytics import YOLO
import cv2
def blur_face(region):
    weights_path = "D:\FramesAndBlur\dataset\YOLO_Model.pt"
    model = YOLO(weights_path)
    res = model(region)
    for r in res:
        boxes = r.boxes
        if boxes:
            for box in boxes:
                bouding_cordinate = box.xyxy
                x1, y1, x2, y2 = ( int(bouding_cordinate[0,0].item()), int(bouding_cordinate[0,1].item()), 
                                int(bouding_cordinate[0,2].item()), int(bouding_cordinate[0,3].item()) )
                x1, y1, x2, y2 = max(x1,0), max(y1, 0), max(x2, 0), max(y2, 0)
                crop = region[y1:y2, x1:x2]
                blurred_region = cv2.GaussianBlur(crop, (25,25), 100)
                region[y1:y2, x1:x2] = blurred_region
    return region