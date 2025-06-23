#!/usr/bin/env python3
import time
import cv2
import numpy as np
import mediapipe as mp
from adafruit_servokit import ServoKit
from picamera2 import Picamera2
from sshkeyboard import listen_keyboard, stop_listening
import threading

# Constants
RES_W, RES_H = 640, 480

# ServoKit setup
kit = ServoKit(channels=16)
pan_angle = 90
tilt_angle = 90
kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

# Keyboard listener
loop = True
def press(key):
    global loop
    if key == 'q':
        loop = False
def release(key): pass
def input_keyboard():
    listen_keyboard(on_press=press, on_release=release, delay_second_char=0.001)
keyboard_thread = threading.Thread(target=input_keyboard, daemon=True)
keyboard_thread.start()

# Drawing color
hand_color = (0, 255, 0)

# Clamp angle
def clamp_angle(a):
    return max(0, min(180, a))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.4
)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format":"XRGB8888", "size":(RES_W, RES_H)}))
picam2.start()

# Setup for servo region calculation
regions_initialized = False
XC = XL = XR = XP = YC = YT = YB = YP = None

fps_avg_len = 200
frame_rate_buffer = []
avg_frame_rate = 0.0

while loop:
    t_start = time.perf_counter()

    # Grab frame from camera
    frame_bgra = picam2.capture_array()
    frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # Set up tracking regions on first frame
    if not regions_initialized:
        h, w = frame.shape[:2]
        XC = w / 2
        YC = h * 7/16
        XR = XC - 100
        XL = XC + 100
        XP = w / 16
        YT = YC - 100
        YB = YC + 100
        YP = h / 16
        regions_initialized = True

    # Hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    best_box = None
    best_score = 0.0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            score = handedness.classification[0].score
            if score >= 0.6:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                best_box = (x_min, y_min, x_max, y_max)
                best_score = score
                break

    # Servo & drawing
    if best_box is not None:
        xmin, ymin, xmax, ymax = best_box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), hand_color, 2)
        label = f'hand: {int(best_score*100)}%'
        (labelW, labelH), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(ymin, labelH + 10)
        cv2.rectangle(frame, (xmin, label_y - labelH - 10),
                      (xmin + labelW, label_y + baseLine - 10), hand_color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        if center_x < XL or center_x > XR:
            pan_angle = int(round(pan_angle - (center_x - XC) / XP))
        if center_y < YT or center_y > YB:
            tilt_angle = int(round(tilt_angle + (center_y - YC) / YP))
        pan_angle = clamp_angle(pan_angle)
        tilt_angle = clamp_angle(tilt_angle)
        kit.servo[0].angle = pan_angle
        kit.servo[1].angle = tilt_angle

    object_count = 1 if best_box is not None else 0
    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Hands: {object_count}', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow('Pan-Tilt Hands Tracking', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

    # FPS calculation
    t_stop = time.perf_counter()
    if (t_stop - t_start) > 0:
        frame_rate_buffer.append(1.0 / (t_stop - t_start))
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

# Cleanup
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
picam2.stop()
cv2.destroyAllWindows()
stop_listening()
hands.close()
