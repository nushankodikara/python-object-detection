import math
from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np
from sort import *

print(torch.backends.mps.is_available())

model = YOLO("yolo-weights/yolov8n.pt")

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

video_path = "videos/people.mp4"

cap = cv2.VideoCapture(video_path)
cap.set(3, 640)
cap.set(4, 480)

# Tracking
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.4)

tracker_up_line = [(151,251),(282,225)]
tracker_down_line = [(724,598),(659,629)]

people_up_ids = []
people_down_ids = []

search_area = [(3,5),(3,190),(224,715),(917,716),(485,3)]

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

# Program Loop

while True:
    success, frame = cap.read()
    if not success:
        break
    
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, np.array([search_area], np.int32), (255, 255, 255))
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

            if currentClass in ["person"] and conf > 0.3:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=8, rt=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    results_tracker = tracker.update(detections)

    cv2.line(frame, (tracker_up_line[0][0], tracker_up_line[0][1]), (tracker_up_line[1][0], tracker_up_line[1][1]), (0, 0, 255), 2)
    cv2.line(frame, (tracker_down_line[0][0], tracker_down_line[0][1]), (tracker_down_line[1][0], tracker_down_line[1][1]), (0, 0, 255), 2)

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
            if line_intersection(edge, (tracker_up_line[0][0], tracker_up_line[0][1], tracker_up_line[1][0], tracker_up_line[1][1])):
                cv2.putText(frame, f"Detected Up", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if track_id not in people_up_ids:
                    people_up_ids.append(track_id)
        
            if line_intersection(edge, (tracker_down_line[0][0], tracker_down_line[0][1], tracker_down_line[1][0], tracker_down_line[1][1])):
                cv2.putText(frame, f"Detected Down", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if track_id not in people_down_ids:
                    people_down_ids.append(track_id)

    cv2.putText(frame, f"People Up: {len(people_up_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"People Down: {len(people_down_ids)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
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