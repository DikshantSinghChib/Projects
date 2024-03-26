import random
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()

class_list = data.split("\n")
my_file.close()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Generate random colors for class list
detection_colors = []
detected_vehicles =[]
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

frame_wid = 640
frame_hyt = 480
line_pos = 550
offset = 6
count = int(0)
vehicle_counts = []
all_confidence_scores = []
all_box_sizes = []

cap = cv2.VideoCapture("video1.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly 

    if not ret:
        print("Can't receive frame")
        break

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    cv2.line(frame, (25, line_pos), (1200, line_pos), (255, 127, 0), 3)
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],3,)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",(int(bb[0]), int(bb[1]) - 10),
                font,1,(255, 255, 255),2,)
            
            center = center_handle(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
            detection_colors.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255))

    # Check for vehicles that crossed the line
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]
        bb = box.xyxy.numpy()[0]
        x1 = int(bb[0])
        y1 = int(bb[1])
        x2 = int(bb[2])
        y2 = int(bb[3])

        # Calculate the center of the bounding box
        center = center_handle(x1, y1, x2 - x1, y2 - y1)

        x, y = center

        if y > (line_pos - offset) and y < (line_pos + offset):
            count = count + 1
            detected_vehicles.append((x, y))

    vehicle_counts.append(count)

    confidence_scores = [box.conf.numpy()[0] for box in detect_params[0].boxes]
    all_confidence_scores.extend(confidence_scores)

    box_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in detect_params[0].boxes.xyxy.numpy()]
    all_box_sizes.extend(box_sizes)

    cv2.putText(frame, f"Vehicles Count : {count}", (460, 70), font, 2, (0, 0, 255), 5)
    cv2.imshow("ObjectDetection", frame)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()

if all_confidence_scores:
    plt.hist(all_confidence_scores, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Detection Confidence Scores')
    plt.show()

plt.plot(vehicle_counts, color='green', marker='o')
plt.xlabel('Frame Number')
plt.ylabel('Number of Vehicles')
plt.title('Vehicle Count Over Time')
plt.show()

if all_box_sizes:
    plt.hist(all_box_sizes, bins=20, color='orange', alpha=0.7)
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bounding Box Sizes')
    plt.show()

cap.release()
