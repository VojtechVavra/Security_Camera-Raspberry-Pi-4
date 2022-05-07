#!/bin/bash

SERVICE="start_virt_env_and_program"
if ps -fA | grep python | grep TFLite_detection_cam5.py
then
    echo "$SERVICE is running"   
else
    echo "$SERVICE not running"
    bash /home/pi/Projects/start_virt_env_and_program.sh &
    # uncomment to start nginx if stopped
    # systemctl start nginx
fi

SERVICE="convert_h264_to_mp4_compressed.py"
#if pgrep -x "$SERVICE" >/dev/null
if ps -fA | grep python | grep convert_h264_to_mp4_compressed.py
then
    echo "$SERVICE is running"
else
    echo "$SERVICE not running"
    yes | python3 /home/pi/Projects/Python/convert_h264_to_mp4_compressed.py &
    # uncomment to start nginx if stopped
    # systemctl start nginx
fi
