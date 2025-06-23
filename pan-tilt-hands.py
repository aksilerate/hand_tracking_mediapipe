#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Servo control imports
from adafruit_servokit import ServoKit
from sshkeyboard import listen_keyboard, stop_listening
import threading

# --------------------------------------------------------------------------------------------------
# Argument parsing (same as yolo_detect.py)
# --------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Pan-Tilt Hand Tracker (uses YOLO + ServoKit)")
parser.add_argument(
    '--model',
    help='Path to YOLO model file (e.g. "runs/detect/train/weights/best.pt")',
    required=True
)
parser.add_argument(
    '--source',
    help='Image source: file ("test.jpg"), folder ("test_dir"), video ("testvid.mp4"), or camera ("usb0")',
    required=True
)
parser.add_argument(
    '--thresh',
    help='Minimum confidence threshold for displaying detected objects (default: 0.5)',
    type=float,
    default=0.5
)
parser.add_argument(
    '--resolution',
    help='Resolution to display inference results (e.g. "640x480"); if omitted, uses source resolution',
    default=None
)
parser.add_argument(
    '--record',
    help='Record inference to "demo1.avi" (only for video/USB); requires --resolution',
    action='store_true'
)
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# --------------------------------------------------------------------------------------------------
# Validate model path
# --------------------------------------------------------------------------------------------------
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(1)

# --------------------------------------------------------------------------------------------------
# Load YOLO model
# --------------------------------------------------------------------------------------------------
model = YOLO(model_path, task='detect')
labels = model.names  # dictionary mapping class IDs to names

# --------------------------------------------------------------------------------------------------
# Decide source type (copy‐paste from yolo_detect.py)
# --------------------------------------------------------------------------------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'ERROR: File extension {ext} not supported.')
        sys.exit(1)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source.replace('usb',''))
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source.replace('picamera',''))
else:
    print(f'ERROR: Input {img_source} is invalid. Please try again.')
    sys.exit(1)

# --------------------------------------------------------------------------------------------------
# Parse resolution if provided
# --------------------------------------------------------------------------------------------------
resize = False
if user_res:
    resize = True
    try:
        resW, resH = map(int, user_res.split('x'))
    except:
        print('ERROR: Resolution must be "WIDTHxHEIGHT" (e.g. "640x480").')
        sys.exit(1)

# --------------------------------------------------------------------------------------------------
# Set up recording, if requested
# --------------------------------------------------------------------------------------------------
if record:
    if source_type not in ['video','usb']:
        print('ERROR: Recording only works for video or USB camera sources.')
        sys.exit(1)
    if not user_res:
        print('ERROR: Please specify --resolution to record.')
        sys.exit(1)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        record_fps,
        (resW, resH)
    )

# --------------------------------------------------------------------------------------------------
# Initialize video source
# --------------------------------------------------------------------------------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for f in glob.glob(os.path.join(img_source, '*')):
        _, e = os.path.splitext(f)
        if e in img_ext_list:
            imgs_list.append(f)
elif source_type in ['video','usb']:
    if source_type == 'video':
        cap_arg = img_source
    else:
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format":"XRGB8888", "size":(resW,resH)}))
    cap.start()

# --------------------------------------------------------------------------------------------------
# ServoKit setup (borrowed from pan-tilt-face.py)
# --------------------------------------------------------------------------------------------------
kit = ServoKit(channels=16)
pan_angle = 90
tilt_angle = 90
kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

# --------------------------------------------------------------------------------------------------
# Keyboard listener (optional; allows you to press 'q' to quit)
# --------------------------------------------------------------------------------------------------
loop = True
release_a = release_d = release_w = release_s = False

def press(key):
    global loop
    if key == 'q':
        loop = False

def release(key):
    # Not used for autonomous tracking, but we define it so keyboard won't error.
    pass

def input_keyboard():
    listen_keyboard(on_press=press, on_release=release, delay_second_char=0.001)

keyboard_thread = threading.Thread(target=input_keyboard, daemon=True)
keyboard_thread.start()

# --------------------------------------------------------------------------------------------------
# Colors for drawing (just use a fixed color for “hand”)
# --------------------------------------------------------------------------------------------------
hand_color = (0, 255, 0)  # bright green

# --------------------------------------------------------------------------------------------------
# Constants: which YOLO classes correspond to “hands”
# --------------------------------------------------------------------------------------------------
HAND_CLASSES = {"yourright","yourleft","myright","myleft"}

# --------------------------------------------------------------------------------------------------
# Helper: clamp angle to [0, 180]
# --------------------------------------------------------------------------------------------------
def clamp_angle(a):
    return max(0, min(180, a))

# --------------------------------------------------------------------------------------------------
# Frame‐processing loop
# --------------------------------------------------------------------------------------------------
img_count = 0
fps_avg_len = 200
frame_rate_buffer = []
avg_frame_rate = 0.0

# We'll only determine region thresholds once we know the actual resized frame shape.
# So we keep a flag to say “haven't initialized regions yet.”
regions_initialized = False

# Variables to store region thresholds:
XC = XL = XR = XP = YC = YT = YB = YP = None

while True:
    t_start = time.perf_counter()

    # 1) Read one frame
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
        if frame is None:
            print('ERROR: Could not read image. Exiting.')
            break
    elif source_type == 'video' or source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Reached end of stream or cannot grab frame. Exiting.')
            break
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('ERROR: Could not grab from picamera.')
            break

    # 2) Resize if requested
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # 3) On the very first frame, set up region thresholds exactly like pan-tilt-face.py:
    if not regions_initialized:
        h, w = frame.shape[:2]
        # center points
        XC = w / 2
        YC = h * 7/16
        # left/right margins (33/32 and 31/32 of XC)
        XR = XC -100
        XL = XC +100
        # pan proportional divisor
        XP = w / 16
        # top/bottom margins for tilt
        YT = YC -100
        YB = YC +100
        # tilt proportional divisor
        YP = h / 16

        regions_initialized = True

    # 4) Run YOLO on the frame
    results = model(frame, verbose=False)
    detections = results[0].boxes  # list of detected boxes

    # 5) Among all detections, pick only those classes in HAND_CLASSES, and track the one with highest confidence.
    best_conf = -1.0
    best_box = None  # (xmin,ymin,xmax,ymax)

    for det in detections:
        class_id = int(det.cls.item())
        name = labels[class_id]
        conf = float(det.conf.item())
        if name in HAND_CLASSES and conf >= min_thresh:
            # get coordinates
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            if conf > best_conf:
                best_conf = conf
                best_box = xyxy  # numpy array [xmin, ymin, xmax, ymax]

    # 6) If we found at least one “hand,” move the servos to center it.
    if best_box is not None:
        xmin, ymin, xmax, ymax = best_box
        # draw the chosen “hand” box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), hand_color, 2)
        label = f'hand: {int(best_conf*100)}%'
        (labelW, labelH), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(ymin, labelH + 10)
        cv2.rectangle(frame, (xmin, label_y - labelH - 10),
                      (xmin + labelW, label_y + baseLine - 10), hand_color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # compute the box center
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # adjust pan: if box center is left of XL or right of XR
        if center_x < XL or center_x > XR:
            pan_angle = int(round(pan_angle - (center_x - XC) / XP))

        # adjust tilt: if box center is above YT or below YB
        if center_y < YT or center_y > YB:
            tilt_angle = int(round(tilt_angle + (center_y - YC) / YP))

        # clamp angles
        pan_angle = clamp_angle(pan_angle)
        tilt_angle = clamp_angle(tilt_angle)

        # update servos
        kit.servo[0].angle = pan_angle
        kit.servo[1].angle = tilt_angle

    # 7) Draw frame‐rate and total-“hand”-count info
    object_count = 1 if best_box is not None else 0
    if source_type in ['video','usb','picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Hands: {object_count}', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 8) Show the frame
    cv2.imshow('Pan-Tilt Hands Tracking', frame)
    if record:
        recorder.write(frame)

    # 9) Handle keypress
    if source_type in ['image','folder']:
        key = cv2.waitKey() & 0xFF
    else:
        key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break

    # 10) Compute FPS
    t_stop = time.perf_counter()
    if (t_stop - t_start) > 0:
        frame_rate_buffer.append(1.0 / (t_stop - t_start))
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

# --------------------------------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video','usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
stop_listening()
