import cv2 
import numpy as np 
from counter import *

input_name = input("Type the name of the file in input folder : ")
cap = cv2.VideoCapture("input/"+input_name)
x1 = int(input("Type the x-coordinate of first point of line : "))
y1 = int(input("Type the y-coordinate of first point of line : "))
x2 = int(input("Type the x-coordinate of second point of line : "))
y2 = int(input("Type the y-coordinate of second point of line : "))


if(cap.isOpened()==False):
	print("Error reading video !")
ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
f_height, f_width, _ = frame.shape
line = [(x1//2,y1//2),(x2//2, y2//2)]
#line = [(0,int(f_height/1.5)),(f_width-1, int(f_height/1.5))]

object_counter = Counter(frame, line)

result = cv2.VideoWriter('output1.avi',  cv2.VideoWriter_fourcc('M','J','P','G'), 10, (f_width,f_height)) 

while ret:
	object_counter.count(frame)
	frame_ = object_counter.show_frame()
	# cv2.imshow('Frame', frame_) 
	# cv2.waitKey(25)
	result.write(frame_) 
	print(object_counter.vehicle_count)
	ret, frame = cap.read()
	if ret:
  		frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)

cap.release() 
result.release()
cv2.destroyAllWindows() 