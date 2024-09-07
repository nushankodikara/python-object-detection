import cv2
from ultralytics import YOLO

model = YOLO("yolo-weights/yolov10x.pt")
results = model("chapter 5 - Running YOLO/images/3.jpeg")
print(results)

cv2.imshow("bus", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()