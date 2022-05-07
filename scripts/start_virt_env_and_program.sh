#!/bin/bash

# start virtual env (environment)
source ~/Projects/Python/tflite/tflite-env/bin/activate

# start application
cd ~/Projects/Python/tflite/object_detection
#python3 TFLite_detection_cam.py --modeldir=coco_ssd_mobilenet_v1
#python3 TFLite_detection_cam2.py --resolution 1280x720 --modeldir=coco_ssd_mobilenet_v1
#python3 TFLite_detection_cam2.py --resolution 1280x720 --modeldir=coco_ssd_mobilenet_v1
#python3 TFLite_detection_cam3.py --resolution 1280x720 --modeldir=coco_ssd_mobilenet_v1
#python3 TFLite_detection_cam4.py --resolution 1280x720 --modeldir=coco_ssd_mobilenet_v1
#python3 TFLite_detection_cam4.py --resolution 1640x1232 --modeldir=coco_ssd_mobilenet_v1

python3 TFLite_detection_cam5.py --resolution 1920x1080 --modeldir=coco_ssd_mobilenet_v1
#python3 TFLite_detection_cam5.py --resolution 1920x1080 --modeldir=coco_ssd_mobilenet_v1


# stop virtual env
# deactivate
