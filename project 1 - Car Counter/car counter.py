import math
from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np  # Import numpy
print(torch.backends.mps.is_available())

model = YOLO("yolo-weights/yolov10n.pt")

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

cap = cv2.VideoCapture("videos/cars.mp4")
cap.set(3, 640)
cap.set(4, 480)

# Program Loop

while True:
    success, frame = cap.read()
    if not success:
        break

    # Create a mask with the same dimensions as the frame, initialized to zero (black)
    mask = np.zeros_like(frame)
    # Define the polygon coordinates
    polygon = np.array([[252, 359], [730, 438], [683, 188], [501, 186]], np.int32)
    # Draw the polygon on the mask with white color
    cv2.fillPoly(mask, [polygon], (255, 255, 255))

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, mask)

    results = model(masked_frame, stream=True, device="mps")
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass in ["car", "motorbike", "bus", "truck"]:
                # Draw detections on the original frame
                cvzone.cornerRect(frame, (x1, y1, w, h), l=8)
                conf = math.ceil((box.conf[0]*100))/100
                cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=16, colorR=(0, 255, 0))

    cv2.imshow("frame", frame)  # Show the original frame with detections
    cv2.waitKey(1)

# Checking Coordinates
# success, frame = cap.read()  # Read the first frame
# if success:
#     cv2.imshow("First Frame", frame)

#     def get_coordinates(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             print(f"Coordinates: ({x}, {y})")

#     cv2.setMouseCallback("First Frame", get_coordinates)

#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

# cv2.destroyAllWindows()
# cap.release()