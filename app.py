# app.py
import cv2 as cv
import numpy as np
import mediapipe as mp
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

# create flask app and socket
app = Flask(__name__)
socket = SocketIO(app)

@app.route("/")
def home():
    return render_template('frontend.html')

# webcam video capture
vid = cv.VideoCapture(0)

# mediapipe vision setup
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HandLandmark = mp_hands.HandLandmark

# create landmarker with support for 2 hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=.5)

# get orientation from hand for scaling grabbed objects. returns a rotation matrix
def extract_orientation(hand_landmarks):
    points = hand_landmarks.landmark
    wrist = points[HandLandmark.WRIST]
    index = points[HandLandmark.INDEX_FINGER_MCP]
    pinky = points[HandLandmark.PINKY_MCP]

    v_index = np.array([index.x - wrist.x, index.y - wrist.y, index.z - wrist.z], dtype=np.float32)
    v_pinky = np.array([pinky.x - wrist.x, pinky.y - wrist.y, pinky.z - wrist.z], dtype=np.float32)

    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    x_axis = _normalize(v_index)
    palm_normal = np.cross(v_index, v_pinky)
    z_axis = _normalize(palm_normal)

    if x_axis is None or z_axis is None:
        return np.eye(3, dtype=np.float32)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis)
    if y_axis is None:
        return np.eye(3, dtype=np.float32)

    return np.stack([x_axis, y_axis, z_axis], axis=1)

# get hand pose data from webcam
def get_hand_data():
    while True:
        frame_landmarks = []
        rotation_matrices = []

        # process frame
        ret, frame = vid.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        mp_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(mp_image)

        # record pose for each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cur_hand_landmarks = []
                for landmark in hand_landmarks.landmark:
                    coords = (landmark.x, landmark.y, landmark.z)
                    cur_hand_landmarks.append(coords)

                rotation_matrix = extract_orientation(hand_landmarks)
                rotation_matrices.append(rotation_matrix.tolist())
                frame_landmarks.append(cur_hand_landmarks)

            socket.emit('new_data', {'frame_landmarks': frame_landmarks, 'rotation_matrices': rotation_matrices})

        # cv.imshow("VIDEO", frame)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cv.destroyAllWindows()

if __name__=='__main__':
    threading.Thread(target=get_hand_data, daemon=True).start()
    app.run()
