import numpy as np
import time
import cv2
from PIL import ImageGrab

LABELS_FILE='/home/da/Desktop/object-detection-opencv-master/yolov3.txt'
CONFIG_FILE='/home/da/Desktop/yolo3-tiny.cfg'
WEIGHTS_FILE='/home/da/Desktop/yolov3-tiny.weights'

CONFIDENCE_THRESHOLD=0
H=None
W=None
LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

lt = time.time()
fps = 0

while True:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0,40,800,640))), cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    if W is None or H is None:
        (H, W) = image.shape[:2]
    layerOutputs = net.forward(ln)

    fps +=1
    if (time.time() - lt) >= 1:
        print(fps)
        fps = 0
        lt = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detections in output:
            scores = detections[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)

    cv2.imshow("output", cv2.resize(image,(800, 600)))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
