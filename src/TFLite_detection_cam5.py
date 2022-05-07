######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
#
# Modified by: Shawn Hymel
# Date: 09/22/20
# Description:
# Added ability to resize cv2 window and added center dot coordinates of each detected object.
# Objects and center coordinates are printed to console.

# 4
"""
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
#import timelapse # my extension file
import io
import time
import picamera
import picamera.array

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    # "Camera object that controls video streaming from the Picamera"
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'h264'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False
        self.recording = False
        # Define the duration (in seconds) of the video capture here
        self.capture_duration = 20

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return
            if self.recording == True:
                continue
            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

    def record(self, time_record):
        self.recording = True
        print("Start recording")
        self.start_time = time.time()
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  #('H','2','6','4')  #('F','M','P','4')  #('M','J','P','G')   # (*'XVID')
        self.now = datetime.now()
        # dd/mm/YY H:M:S
        self.dt_string = self.now.strftime('%Y-%m-%d_%H-%M-%S')
        #d1 = today.strftime("%d/%m/%Y")
        self.timeFormat = '%Y-%m-%d %H:%M:%S'
        # ('img{timestamp:%Y-%m-%d-%H-%M}.jpg'):
        self.filename = "/home/pi/Videos/" + self.dt_string + ".avi"
        self.out = cv2.VideoWriter(self.filename, self.fourcc, 20.0, (1920,  1080))   # (640,  480)

        if self.stream.isOpened():
            while( int(time.time() - self.start_time) < self.capture_duration ):
                self.ret, self.frame = self.stream.read()
                print("Start recording")
                if not self.ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                #self.frame = cv2.flip(self.frame, 0)
                # write the flipped frame
                self.out.write(self.frame)

                cv2.imshow("record", self.frame)


                # check to see if a key was pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            print("End recording")
            self.recording = False

    def record2(self, time_record):
        self.stop()
        #picamera.PiCamera()*
        with picamera.PiCamera() as self.camera:
                self.camera.resolution = (1920, 1080)    # (1280, 720) (640, 480)
                #print("camera quality " + camera.quality)
                #print("camera quality " + camera.quality)
                # datetime object containing current date and time
                self.now = datetime.now()
                # dd/mm/YY H:M:S
                self.dt_string = self.now.strftime("%d%m%Y_%H%M%S")
                #d1 = today.strftime("%d/%m/%Y")
                self.timeFormat = '%Y-%m-%d %H:%M:%S'
                # ('img{timestamp:%Y-%m-%d-%H-%M}.jpg'):
                self.filename = "/home/pi/Videos/" + self.now.strftime('%Y-%m-%d_%H-%M-%S') + ".h264"
                self.camera.annotate_background = picamera.Color('black')
                self.camera.annotate_text = self.now.strftime(time_record)
                #camera.bitrate = 25000000 # 25 Mbps; default: 17000000 (17Mbps)

                print("Start recording...")
                self.camera.start_recording(self.filename, format='h264', quality=20, bitrate=25000000, splitter_port=1)

                #camera.start_recording(filename)
                self.camera.wait_recording(time_record)
                #for i in range(2, 11):
                #    camera.split_recording('%d.h264' % i)
                #    camera.wait_recording(5)
                self.camera.stop_recording()
                print("Done recording")
        self.start()

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

# Create window
#cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
"""




# 5
import random
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
#import timelapse # my extension file
import io
import picamera
from picamera import Color
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


def object_detection(stream):
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

			if False:
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

            # Draw framerate in corner of frame
            #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            # All the results have been drawn on the frame, so it's time to display it.


			# Calculate framerate
			#t2 = cv2.getTickCount()
			#time1 = (t2-t1)/freq
			#frame_rate_calc= 1/time1

			# Press 'q' to quit
			#if cv2.waitKey(1) == ord('q'):
			#	break

			return object_name == "person"


def detect_motion(camera):
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
		result = object_detection(stream)

		# Once motion detection is done, make the prior image the current
		prior_image = current_image
		return result


# https://buildmedia.readthedocs.org/media/pdf/picamera/latest/picamera.pdf str 46 - 50

with picamera.PiCamera() as camera:
    camera.resolution = (imW, imH)  # (1920, 1080) # (1640x1232) (1280, 720)
    # camera = PiCamera(resolution=(1920, 1080), framerate=15,sensor_mode=5)
    camera.framerate = 22
    #camera.exposure_mode = 'night'
    # start and wait for camera setup
    time.sleep(1)
    stream = picamera.PiCameraCircularIO(camera, seconds=1) # uncomment for pre recording save

    """quality - Specifies the quality that the encoder should attempt to maintain. For the 'h264' format,
    use values between 10 and 40 where 10 is extremely high quality, and 40 is extremely low (20-25 is usually
    a reasonable range for H.264 encoding). For the mjpeg format, use JPEG quality values between 1 and 100
    (where higher values are higher quality). Quality 0 is special and seems to be a “reasonable quality” default."""
    # 1-40 for H.264 recordings with lower values indicating higher quality
    camera.start_recording(stream, format='h264', quality=23) # originally without quality now 21 ;  # uncomment for pre recording save

    try:
        while True:
            camera.wait_recording(1)
            if detect_motion(camera):
                print('Motion detected!')
                stream.clear()
				# As soon as we detect motion, split the recording to
				# record the frames "after" motion
                file_path = "/home/pi/Videos/"
				# datetime object containing current date and time

                now = datetime.now()
                timeNow = now.strftime('%Y-%m-%d_%H-%M-%S')
                filename = timeNow + ".h264" # + "_after.h264"
				# Values can be between 1 (highest quality)
				# and 40 (lowest quality). 20 - 23 optional

                camera.annotate_background = Color("black")
                camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #camera.split_recording(file_path + filename, format='h264', quality=23) # uncomment for pre recording save
                # Write the 10 seconds "before" motion to disk as well
                #write_video(stream) # uncomment for pre recording save
                # Wait until motion is no longer detected, then split
                # recording back to the in-memory circular buffer

                #camera.start_recording(file_path + filename, format='h264', quality=25)
                camera.split_recording(file_path + filename, format='h264', quality=23)
                camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                camera.wait_recording(0.2)
                while detect_motion(camera):
                    print('while detect_motion(camera):')
                    start = datetime.now()
                    while (datetime.now() - start).seconds < 5:
                        print((datetime.now() - start).seconds)
                        camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        camera.wait_recording(0.2)
                        #camera.wait_recording(5)

                print((datetime.now() - start).seconds)
                #camera.stop_recording()
                print('Motion stopped!')
                camera.split_recording(stream)
                lock_and_write(filename) # "after.h264"
                #camera.split_recording(stream)
            else:
                print("no detection")

    finally:
        camera.stop_recording()


"""if object_name == "person":
                    camera.resolution = (1920, 1080)    # (1280, 720) (640, 480)
                    # datetime object containing current date and time
                    now = datetime.now()
                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d%m%Y_%H%M%S")
                    #d1 = today.strftime("%d/%m/%Y")
                    timeNow = now.strftime('%Y-%m-%d_%H-%M-%S')
                    # ('img{timestamp:%Y-%m-%d-%H-%M}.jpg'):
                    filename = "/home/pi/Videos/" + timeNow + ".h264"
                    #camera.annotate_background = picamera.Color('black')
                    camera.annotate_text = filename
                    #camera.bitrate = 25000000 # 25 Mbps; default: 17000000 (17Mbps)
                    print("Start recording...")
                    camera.start_recording(filename, format='h264', quality=20, bitrate=25000000, splitter_port=1)
                    #camera.start_recording('/home/pi/Videos/foo22.h264')
                    camera.wait_recording(15)
                    #camera.capture('foo.jpg', use_video_port=True)
                    #camera.wait_recording(10)
                    camera.stop_recording()
                    print("End recording...")
                    break
"""

############################################################
"""
with picamera.PiCamera() as camera:
    camera.resolution = (1920, 1080) #(1280, 720)
    camera.framerate = 15        # 20

    # start and wait for camera setup
    #camera.start_preview()
    time.sleep(2)
    try:
    while True:
        #camera.resolution = (1280, 720)
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Capture using jpeg
        stream = io.BytesIO()
        start = time.time()
        camera.capture(stream, format='jpeg')
        # Construct a numpy array from the stream
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        #image = cv2.imdecode(data, 1)
        #cv2.imwrite('jpegnumpy.png',image)
        print("Time with jpeg+stream+numpy = %.4f" % (time.time()-start))

        # Grab frame from video stream
        #frame1 = videostream.read() # uncomment
        #camera.capture(image, 'bgr')

        # Acquire frame and resize to expected shape [1xHxWx3]
        #frame = frame1.copy() # uncomment
        #image = cv2.imdecode(data, 1)
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
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        print(classes[0])
        # Loop over all detections and draw detection box if confidence is above minimum threshold
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

                # Draw label
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
                    camera.resolution = (1920, 1080)    # (1280, 720) (640, 480)
                    # datetime object containing current date and time
                    now = datetime.now()
                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d%m%Y_%H%M%S")
                    #d1 = today.strftime("%d/%m/%Y")
                    timeNow = now.strftime('%Y-%m-%d_%H-%M-%S')
                    # ('img{timestamp:%Y-%m-%d-%H-%M}.jpg'):
                    filename = "/home/pi/Videos/" + timeNow + ".h264"
                    #camera.annotate_background = picamera.Color('black')
                    camera.annotate_text = filename
                    #camera.bitrate = 25000000 # 25 Mbps; default: 17000000 (17Mbps)
                    print("Start recording...")
                    camera.start_recording(filename, format='h264', quality=20, bitrate=25000000, splitter_port=1)
                    #camera.start_recording('/home/pi/Videos/foo22.h264')
                    camera.wait_recording(15)
                    #camera.capture('foo.jpg', use_video_port=True)
                    #camera.wait_recording(10)
                    camera.stop_recording()
                    print("End recording...")
                    break

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame) # show frame

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    finally:
        camera.stop_recording()
# Clean up
cv2.destroyAllWindows()
#videostream.stop()

"""
