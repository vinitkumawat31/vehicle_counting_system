import cv2 
import numpy as np 

from yolo import *
from vehicle import *

class Counter:
	def __init__(self, frame, line):
		self.frame = frame 
		self.detection_interval = 5
		self.line = line
		self.vehicles = {}
		self.f_height, self.f_width, _ = self.frame.shape
		self.frame_count = 1 
		self.vehicle_count = 0
		self.net, self.labels = load_model()
		_bounding_boxes, _classes, _confidences = get_boxes(self.frame, self.net, self.labels)
		self.vehicles = add_new_vehicles(_bounding_boxes, _classes, _confidences, self.vehicles, self.frame)

	def count(self, frame):
		self.frame = frame
		for key,veh in self.vehicles.copy().items():
			veh = update_vehicle_tracker(veh,self.frame)
			self.vehicles[key] = veh
		for key,veh in self.vehicles.copy().items():
			cross = check_cross(key, veh, self.line)
			if cross == 1:
				veh.crossed = True
			self.vehicle_count = self.vehicle_count + cross
			if veh.track_fails >= 3:
				self.vehicles.pop(key)

		if self.frame_count % self.detection_interval == 0:
			_bounding_boxes, _classes, _confidences = get_boxes(frame, self.net, self.labels)
			self.vehicles = add_new_vehicles(_bounding_boxes, _classes, _confidences, self.vehicles, self.frame)
			self.vehicles = remove_duplicates(self.vehicles)
		self.frame_count = self.frame_count + 1

	def show_frame(self):
		frame = self.frame.copy()
		color = (0,255,0)
		for key,veh in self.vehicles.items():
			box = veh.box
			(x, y) = (box[0], box[1])
			(w, h) = (box[2], box[3])
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{:.4f}".format(veh.score)
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

		text = "Count: {}".format(self.vehicle_count)
		cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0,0,255), 2)
		cv2.line(frame, self.line[0], self.line[1], color, 3)
		return frame