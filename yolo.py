import numpy as np
import time
import cv2
import os


def load_model():
	labels = open("coco.names").read().strip().split("\n")
	print("loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
	print("YOLO loaded...")
	return net, labels

def get_boxes(image, net, labels):
	(H, W) = image.shape[:2]
	l_names = net.getLayerNames()
	output_layers = [l_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(output_layers)
	end = time.time()
	print("YOLO took {:.3f} seconds".format(end - start))

	vehicles = ["bicycle","car","truck","motorcycle","bus"]
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.5 and labels[classID] in vehicles:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	boxes_ = []
	confidences_ = []
	classes = []
	if(len(idxs)>0):
		for i in idxs.flatten():
			boxes_.append(boxes[i])
			confidences_.append(confidences[i])
			classes.append(classIDs[i])

	return boxes_, classes, confidences_


def show_detection(image, boxes, classes, confidences, labels):
	image_ = image.copy()
	for i in range(0,len(boxes)):
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		color = (0,255,0)
		cv2.rectangle(image_, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(labels[classes[i]], confidences[i])
		cv2.putText(image_, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	return image_



# image = cv2.imread("person.jpg")
# cap = cv2.VideoCapture('input/20201230123300.mp4')
# ret, frame = cap.read()
# image = frame
# net,labels = load_model()

# boxes, classes, confidences = get_boxes(image, net, labels)

# detection = show_detection(image, boxes, classes, confidences, labels)
# cv2.imwrite("detection.jpg", detection) 
  