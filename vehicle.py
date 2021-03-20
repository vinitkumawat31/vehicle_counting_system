import cv2 
import numpy as np 

class Vehicle:
	def __init__(self, _box, _tracker, _score, _label):
		self.box = _box
		self.tracker = _tracker
		self.score = _score
		self.crossed = False
		self.track_fails = 0
		self.detection_fails = 0
		self.label = _label

	def update(self, _box, _score, _tracker):
		self.box = _box
		self.tracker = _tracker
		self.score = _score


def box_overlap(box1, box2):
	x11 = box1[0]
	y11 = box1[1]
	x21 = x11 + box1[2]
	y21 = y11 + box1[3]

	x12 = box2[0]
	y12 = box2[1]
	x22 = x12 + box2[2]
	y22 = y12 + box2[3]

	overlap_x1 = max(x11,x12)
	overlap_y1 = max(y11,y12)
	overlap_x2 = min(x21,x22)
	overlap_y2 = min(y21,y22)

	overlap_w = overlap_x2 - overlap_x1
	overlap_h = overlap_y2 - overlap_y1

	if overlap_h < 0 or overlap_w < 0:
		return 0
	overlap_area = overlap_h*overlap_w
	area = min(box1[2]*box1[3], box2[2]*box2[3])

	return overlap_area / area

def remove_duplicates(vehicles):
	for key1,veh1 in vehicles.copy().items():
		for key2,veh2 in vehicles.copy().items():
			if veh1 != veh2:
				if box_overlap(veh1.box, veh2.box) >= 0.6 and key1 in vehicles.keys():
					vehicles.pop(key1)
	return vehicles

def remove_stray_vehicles(vehicles, matched_keys):
	for key, veh in vehicles.copy().items():
		if key not in matched_keys:
			veh.detection_fails  = veh.detection_fails +1
		if veh.detection_fails > 3 :
			vehicles.pop(key)
	return vehicles

def add_new_vehicles(boxes, classes, confidences, vehicles, frame):
	matched_keys = []

	for i, box in enumerate(boxes):
		c = classes[i]
		score = confidences[i]
		tracker = cv2.TrackerCSRT_create()
		tracker.init(frame, tuple(box))

		match = False

		for key, veh in vehicles.items():
			if box_overlap(veh.box, box) >= 0.6:
				match = True
				if key not in matched_keys:
					matched_keys.append(key)
					veh.detection_fails = 0
				veh.update(box,score,tracker)
				break

		if match == False:
			veh = Vehicle(box, tracker, score, c)
			if len(vehicles.keys())!=0:
				key = max(vehicles.keys())+1
			else:
				key = 1
			vehicles[key] = veh
			matched_keys.append(key)

	vehicles = remove_stray_vehicles(vehicles, matched_keys)
	return vehicles

def update_vehicle_tracker(veh, frame):
	success, box = veh.tracker.update(frame)
	if success:
		veh.track_fails = 0
		veh.update(box, veh.score, veh.tracker)
	else:
		veh.track_fails = veh.track_fails + 1
	return veh

def intersect(line1, line2):
	def get_orientation(p, q, r):
		val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
		if val == 0:
			return 0
		return 1 if val > 0 else 2

	def is_on_segment(p, q, r):
		if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
			return True
		return False

	p1 = line1[0]
	q1 = line1[1]
	p2 = line2[0]
	q2 = line2[1]
	o1 = get_orientation(p1, q1, p2)
	o2 = get_orientation(p1, q1, q2)
	o3 = get_orientation(p2, q2, p1)
	o4 = get_orientation(p2, q2, q1)

	if o1 != o2 and o3 != o4:
		return True
	if o1 == 0 and is_on_segment(p1, p2, q1):
		return True
	if o2 == 0 and is_on_segment(p1, q2, q1):
		return True
	if o3 == 0 and is_on_segment(p2, p1, q2):
		return True
	if o4 == 0 and is_on_segment(p2, q1, q2):
		return True
	return False

def box_line_intersect(box, line):
    x, y, w, h = box
    line1 = [(x, y), (x + w, y)]
    line2 = [(x + w, y), (x + w, y + h)]
    line3 = [(x, y), (x, y + h)]
    line4 = [(x, y + h), (x + w, y + h)]

    if intersect(line1, line) or intersect(line2, line) or intersect(line3, line) or intersect(line4, line):
        return True
    return False

def check_cross(key, veh, line):
	if veh.crossed == True:
		return 0

	if box_line_intersect(veh.box, line):
		return 1
	else:
		return 0