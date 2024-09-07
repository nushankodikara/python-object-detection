import math
from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np
from sort import *

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

# Tracking
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.4)

tracker_line = [(391,302),(722,313)]

car_ids = []

# Program Loop

while True:
    success, frame = cap.read()
    if not success:
        break

    # Create a mask with the same dimensions as the frame, initialized to zero (black)
    mask = np.zeros_like(frame)
    # Define the polygon coordinates
    polygon = np.array([[487, 194], [695, 191], [788, 529], [101, 427]], np.int32)
    # Draw the polygon on the mask with white color
    cv2.fillPoly(mask, [polygon], (255, 255, 255))

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, mask)

    results = model(masked_frame, stream=True, device="mps")
    
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            conf = math.ceil((box.conf[0]*100))/100

            if currentClass in ["car", "motorbike", "bus", "truck"] and conf > 0.3:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=8, rt=1)
                # cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=16, colorR=(0, 255, 0))
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    results_tracker = tracker.update(detections)

    cv2.line(frame, (tracker_line[0][0], tracker_line[0][1]), (tracker_line[1][0], tracker_line[1][1]), (0, 0, 255), 2)

    def line_intersection(line1, line2):
        """Check if line1 intersects with line2"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate determinants
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return False  # lines are parallel

        # Calculate the intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        # Check if intersection is within the line segments
        return 0 <= t <= 1 and 0 <= u <= 1

    for result in results_tracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Define the edges of the rectangle
        rect_edges = [
            (x1, y1, x1, y2),  # Left edge
            (x2, y2, x2, y1),  # Right edge
        ]

        # Check each edge for intersection with the tracker line
        for edge in rect_edges:
            if line_intersection(edge, (tracker_line[0][0], tracker_line[0][1], tracker_line[1][0], tracker_line[1][1])):
                cv2.putText(frame, f"Detected", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if track_id not in car_ids:
                    car_ids.append(track_id)
    
    cv2.putText(frame, f"Cars Detected: {len(car_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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