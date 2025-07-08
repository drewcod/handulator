from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math
import threading

# Initialize Flask app and SocketIO
app = Flask(__name__)
socket = SocketIO(app)

@app.route("/")
def home():
    return render_template("frontend2.html")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Gesture detection thresholds and parameters
PINCH_GRAB_THRESHOLD = 0.15  # For 5-finger grab
PINCH_SCALE_THRESHOLD = 0.15  # For pinch scaling (thumb, index, and middle)
RELEASE_VELOCITY_THRESHOLD = 0.15  # Release detection based on finger speed
ANGLE_SENSITIVITY = 5  # Minimum angle change to trigger scaling
SCALE_STEP = 0.05  # Amount to change scale per rotation step

# Global state variables
prev_distances = None
prev_time = None
prev_pinky_angle = None
block_size = 1.0
holding_object = False

def calculate_pinky_angle(pinky, index):
    """Calculate the angle of the pinky relative to the index finger."""
    dx = pinky.x - index.x
    dy = pinky.y - index.y
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def hand_tracking():
    global prev_distances, prev_time, prev_pinky_angle, block_size, holding_object
    cap = cv.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            pinch_grab = False
            pinch_scale = False
            release_detected = False
            pinch_position = None
            grab_position = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]
                    middle = hand_landmarks.landmark[12]
                    ring = hand_landmarks.landmark[16]
                    pinky = hand_landmarks.landmark[20]

                    grab_distances = [
                        abs(thumb.x - f.x) + abs(thumb.y - f.y) + abs(thumb.z - f.z)
                        for f in [index, middle, ring, pinky]
                    ]
                    if all(d < PINCH_GRAB_THRESHOLD for d in grab_distances):
                        pinch_grab = True
                        grab_position = (int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0]))

                    if prev_distances is not None and prev_time is not None:
                        time_diff = time.time() - prev_time
                        max_distance_diff = max(curr - prev for curr, prev in zip(grab_distances, prev_distances))
                        velocity = max_distance_diff / time_diff if time_diff > 0 else 0
                        if velocity > RELEASE_VELOCITY_THRESHOLD:
                            release_detected = True
                            holding_object = False

                    scale_distances = [
                        abs(thumb.x - f.x) + abs(thumb.y - f.y) + abs(thumb.z - f.z)
                        for f in [index, middle]
                    ]
                    if all(d < PINCH_SCALE_THRESHOLD for d in scale_distances):
                        pinch_scale = True
                        pinch_position = (int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0]))

                    if pinch_scale:
                        pinky_angle = calculate_pinky_angle(pinky, index)
                        if prev_pinky_angle is not None:
                            angle_diff = pinky_angle - prev_pinky_angle
                            if angle_diff > ANGLE_SENSITIVITY:
                                block_size += SCALE_STEP
                            elif angle_diff < -ANGLE_SENSITIVITY:
                                block_size = max(0.1, block_size - SCALE_STEP)
                        prev_pinky_angle = pinky_angle

                    prev_distances = grab_distances
                    prev_time = time.time()

                    hand_data = {
                        "frame_landmarks": [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        "block_size": block_size,
                        "holding_object": holding_object,
                        "pinch_grab": pinch_grab,
                        "pinch_scale": pinch_scale,
                        "grab_position": grab_position,
                        "pinch_position": pinch_position
                    }
                    socket.emit('new_data', hand_data)
            #time.sleep(0.03)
            cv.imshow("Video Feed", frame)
            cv.waitKey(1)
    cap.release()

if __name__ == '__main__':
    threading.Thread(target=hand_tracking, daemon=True).start()
    socket.run(app, host="0.0.0.0", port=5000)
