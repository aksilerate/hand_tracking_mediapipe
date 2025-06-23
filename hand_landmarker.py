#!/usr/bin/env python3
"""
MediaPipe Hand Landmarker for Raspberry Pi 5 using Picamera2
Real-time hand landmark detection using camera feed
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Import Picamera2
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

class HandLandmarker:
    def __init__(self,
                 model_complexity=0,
                 min_detection_confidence=0.8,
                 min_tracking_confidence=0.3,
                 max_num_hands=1):
        """
        Initialize MediaPipe Hand Landmarker
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize the hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # For FPS calculation
        self.prev_time = 0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        h, w, _ = frame.shape
        hand_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                score = handedness.classification[0].score  # Confidence score (0.0 - 1.0)

                if score >= 0.6:   # Only draw if confidence > 60%
                    hand_detected = True
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    self._draw_custom_landmarks(frame, hand_landmarks)

                    # Draw green rectangle around hand
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Print and show score
                    cv2.putText(frame, f'Score: {int(score*100)}%', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    break  # Only process the first confident hand

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display hand count (0 or 1)
        num_hands = 1 if hand_detected else 0
        cv2.putText(frame, f'Hands: {num_hands}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def _draw_custom_landmarks(self, frame, hand_landmarks):
        """
        Highlight fingertips
        """
        h, w, _ = frame.shape
        fingertip_indices = [4, 8, 12, 16, 20]
        for idx, lm in enumerate(hand_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            if idx in fingertip_indices:
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run_camera(self, width=640, height=480):
        """
        Run hand detection with Picamera2
        """
        # Initialize Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_video_configuration(main={"format":"XRGB8888", "size":(640,640)}))
        picam2.start()

        print("Hand Landmarker started with Picamera2. Press 'q' to quit, 's' to save screenshot, 'r' to reset.")
        screenshot_counter = 0

        try:
            while True:
                # Capture frame
                frame_rgb = picam2.capture_array()
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                processed = self.process_frame(frame_bgr)
                cv2.imshow('Hand Landmarker - PiCamera2', processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_counter += 1
                    fname = f'hand_detection_{screenshot_counter:03d}.jpg'
                    cv2.imwrite(fname, processed)
                    print(f"Screenshot saved as {fname}")
                elif key == ord('r'):
                    print("Restarting detection...")
                    self.hands.close()
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            picam2.stop()
            cv2.destroyAllWindows()
            self.hands.close()
            print("Camera released and windows closed")


def main():
    print("MediaPipe Hand Landmarker with Picamera2 for Raspberry Pi 5")
    print("=" * 50)
    landmarker = HandLandmarker(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    landmarker.run_camera(width=640, height=480)

if __name__ == "__main__":
    main()

