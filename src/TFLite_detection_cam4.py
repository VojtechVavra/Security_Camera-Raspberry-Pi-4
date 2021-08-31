import io
import random
import picamera
from PIL import Image

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from datetime import datetime, timedelta
# pi camera
import timelapse # my extension file
import io
import picamera
import picamera.array

import fcntl

prior_image = None

lock_file = "/home/pi/Projects/Python/lockfile.lck"

# https://stackoverflow.com/questions/6931342/system-wide-mutex-in-python-on-linux
class Locker:
    def __enter__ (self):
        self.fp = open(lock_file, 'a')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

def lock_and_write(video_record_name):
	with Locker():
		with open(lock_file, 'a') as f: 	# 'a' append, 'w' write
			f.write(video_record_name)
			f.write("\n")
		#print("obtained lock")
		#f = open(lock_file, "a")
		#f.write(video_record_name)
		#f.close()
		#time.sleep(5.0)

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
print(PATH_TO_LABELS)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    print(labels)

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
DEFAULT_FRAMERATE = 30 # 30

# Initialize video stream
# uncomment all videostream
#videostream = VideoStream(resolution=(imW,imH),framerate=DEFAULT_FRAMERATE).start()
time.sleep(1)

def object_detection(stream, start_time):
	# Construct a numpy array from the stream
	data = np.fromstring(stream.getvalue(), dtype=np.uint8)
	
	# Acquire frame and resize to expected shape [1xHxWx3]
	image = cv2.imdecode(data, 1)
	frame = image.copy()
	#frame = image.copy()
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb, (width, height))
	input_data = np.expand_dims(frame_resized, axis=0)

	# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std

	# Perform the actual detection by running the model with the image as input
	interpreter.set_tensor(input_details[0]['index'],input_data)
	interpreter.invoke()

	# Retrieve detection results
	boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
	classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
	scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
	num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

	#print(classes[0])
	
	#Loop over all detections and draw detection box if confidence is above minimum threshold
	for i in range(len(scores)):
		if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
			print("scores " + str(len(scores)))
			print("i = " + str(i))
			print("scores " + str(len(labels)))
                #if videostream.recording == True:
                #    continue
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
			ymin = int(max(1,(boxes[i][0] * imH)))
			xmin = int(max(1,(boxes[i][1] * imW)))
			ymax = int(min(imH,(boxes[i][2] * imH)))
			xmax = int(min(imW,(boxes[i][3] * imW)))

			cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
			
			#Draw label
			object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
			label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'

			labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
			label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
			cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
			cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

			# Draw circle in center
			xcenter = xmin + (int(round((xmax - xmin) / 2)))
			ycenter = ymin + (int(round((ymax - ymin) / 2)))
			cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

                # Print info
			print('Object ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')
			# added
			
			if object_name == "person":
				return True
			elif object_name == "car":
				time_now = datetime.now()
				elapsed = time_now - start_time
				return not (elapsed > timedelta(minutes=5))
			else:
				return False
				
def detect_motion(camera, start_time):
	global prior_image
	stream = io.BytesIO()
	camera.capture(stream, format='jpeg', use_video_port=True)
	stream.seek(0)
	if prior_image is None:
		prior_image = Image.open(stream)
		return False
	else:
		current_image = Image.open(stream)
		# Compare current_image to prior_image to detect motion. This is
		# left as an exercise for the reader!
		#result = random.randint(0, 10) == 0
		result = object_detection(stream, start_time)
        
		# Once motion detection is done, make the prior image the current
		prior_image = current_image
		return result

def write_video(stream):
	# Write the entire content of the circular buffer to disk. No need to
	# lock the stream here as we're definitely not writing to it
	# simultaneously
	file_path = "/home/pi/Videos/"	# "before.h264"
	# datetime object containing current date and time
	now = datetime.now()
	timeNow = now.strftime('%Y-%m-%d_%H-%M-%S')
	filename = timeNow + "_before.h264"
	with io.open(file_path + filename, 'wb') as output:
		for frame in stream.frames:
			if frame.frame_type == picamera.PiVideoFrameType.sps_header:
				stream.seek(frame.position)
				break
		while True:
			buf = stream.read1()
			if not buf:
				break
			output.write(buf)
	# Wipe the circular stream once we're done
	stream.seek(0)
	stream.truncate()
	
	lock_and_write(filename)  # "before.h264"

with picamera.PiCamera(sensor_mode=4) as camera:
	#camera.resolution = (1920, 1080)  #(1920, 1080) # (1640x1232) (1280, 720)
	# camera = PiCamera(resolution=(1920, 1080), framerate=15,sensor_mode=5)
	stream = picamera.PiCameraCircularIO(camera, seconds=10)
	camera.start_recording(stream, format='h264', quality=18) # originally without quality now 21
	time.sleep(1)
	try:
		new_car_time = datetime.now()
		while True:
			camera.wait_recording(1)
			
			if detect_motion(camera, new_car_time):
				print('Motion detected!')
				new_car_time = datetime.now()
				# As soon as we detect motion, split the recording to
				# record the frames "after" motion
				file_path = "/home/pi/Videos/"
				# datetime object containing current date and time
				
				now = datetime.now()
				timeNow = now.strftime('%Y-%m-%d_%H-%M-%S')
				filename = timeNow + "_after.h264"
				# Values can be between 1 (highest quality) 
				# and 40 (lowest quality). 20 - 23 optional
				
				camera.split_recording(file_path + filename, format='h264', quality=21)
				# Write the 10 seconds "before" motion to disk as well
				write_video(stream)
				# Wait until motion is no longer detected, then split
				# recording back to the in-memory circular buffer
				while detect_motion(camera, new_car_time):
					camera.wait_recording(5)	# 1
				print('Motion stopped!')
				
				lock_and_write(filename) # "after.h264"
				camera.split_recording(stream)				
			else:
				pass
				#print("no detection")

	finally:
		camera.stop_recording()
