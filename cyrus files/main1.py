# main.py
import numpy as np 
from flask import Flask, render_template 
from flask_socketio import SocketIO
import cv2 as cv
import mediapipe as mp
import time
import math
import threading


app = Flask(__name__)
socketio = SocketIO(app)

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
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1) as hands:
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
                    socketio.emit('new_data', hand_data)
            #time.sleep(0.03)
            cv.imshow("Video Feed", frame)
            cv.waitKey(1)
    cap.release()


###########################drew and jaya below################


RED = 0xFF0000
GREEN = 0x00FF00
BLUE = 0x0000FF

cube_size = 1.0  # Cube side length
move_step = 0.1  # Movement step size

# Blue (user) cube position
user_pos = np.array([0.0, 0.0, 0.0])

# List of cubes available to grab. Initially, one red cube.
shapes = [
    { 'pos': np.array([2.0, 0.0, 0.0]), 'color': GREEN, 'shape': 'cube' }
]

# Variables to track the currently attached (grabbed) cube.
attached_offset = None  # Offset between user_pos and the attached cube’s initial position.
attached_index = None   # Index into the cubes list of the attached cube.

def cubes_touch(pos1, pos2, size):
    # Return True if the cubes (centered at pos1 and pos2) are close enough to be considered touching.
    return np.all(np.abs(pos1 - pos2) <= size)

@app.route("/")
def home():
    return render_template('frontend.html')

def broadcast_positions():
    shape_data = []
    for i, shape in enumerate(shapes):
        shape_data.append({
            'id': i,
            'pos': shape['pos'].tolist(),
            'color': shape['color'],
            'shape': shape['shape']
        })
    data = {
        'user': user_pos.tolist(),
        'shapes': shape_data,
        'attached_index': attached_index
    }
    socketio.emit('update_shapes', data)

@socketio.on('move')
def handle_move(data):
    global user_pos, attached_offset, shapes
    axis = data.get('axis')
    delta = data.get('delta')
    if axis is None or delta is None:
        return
    user_pos[axis] += delta
    # If a cube is attached, update its position so it stays at the same offset relative to the blue cube.
    if attached_index is not None and attached_offset is not None:
        shapes[attached_index]['pos'] = user_pos + attached_offset
    broadcast_positions()

@socketio.on('attach')
def handle_attach():
    global attached_index, attached_offset, shapes, user_pos
    # Do nothing if already attached.
    if attached_index is not None:
        return

    candidate_index = None
    min_distance = None
    # Look for a candidate cube (red or green) that is touching the blue cube.
    for i, cube in enumerate(shapes):
        if cubes_touch(user_pos, cube['pos'], cube_size):
            distance = np.linalg.norm(user_pos - cube['pos'])
            if candidate_index is None or distance < min_distance:
                candidate_index = i
                min_distance = distance

    if candidate_index is not None:
        # Instead of moving the candidate cube, create a new green cube based on its position.
        candidate_shape = shapes[candidate_index]
        if candidate_index == 0:
            new_cube = {
                'pos': candidate_shape['pos'].copy(),  # Start at the candidate cube’s position.
                'color': candidate_shape['color'],
                'shape': candidate_shape['shape']
            }
            shapes.append(new_cube)
            attached_index = len(shapes) - 1
            attached_offset = new_cube['pos'] - user_pos
        else:
            attached_index = candidate_index
            attached_offset = candidate_shape['pos'] - user_pos
        
    broadcast_positions()

@socketio.on('detach')
def handle_detach():
    global attached_index, attached_offset
    if attached_index is not None:
        attached_index = None
        attached_offset = None
    broadcast_positions()

@socketio.on('changeStationary')
def handle_changeStationary(data):
    global shapes
    shapes[0]['shape'] = data['shape']
    shapes[0]['color'] = data['color']
    broadcast_positions()

if __name__=='__main__':
    threading.Thread(target=hand_tracking, daemon=True).start()
    socketio.run(app, debug=True)
