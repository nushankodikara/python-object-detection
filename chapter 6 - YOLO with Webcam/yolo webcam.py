import math
from ultralytics import YOLO
import cv2
import cvzone
import torch
print(torch.backends.mps.is_available())

model = YOLO("yolo-weights/yolov10l.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    results = model(frame,stream=True, device="mps")
    # print(results)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=16, colorR=(0, 255, 0))

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    
    