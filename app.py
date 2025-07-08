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

# create landmarker with support for 2 hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=.5)

# get hand pose data from webcam
def get_hand_data():
    while True:
        frame_landmarks = []

        # process frame
        ret, frame = vid.read(0)
        frame = cv.flip(frame, 1)
        mp_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        landmarks = hands.process(mp_image)

        # record pose for each detected hand
        if landmarks.multi_hand_landmarks:
            for hand_landmarks in landmarks.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cur_hand_landmarks = []
                for landmark in hand_landmarks.landmark:
                    coords = (landmark.x, landmark.y, landmark.z)
                    cur_hand_landmarks.append(coords)
                frame_landmarks.append(cur_hand_landmarks)
            
            socket.emit('new_data', {'frame_landmarks': frame_landmarks})

        # cv.imshow("VIDEO", frame)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cv.destroyAllWindows()

if __name__=='__main__':
    threading.Thread(target=get_hand_data, daemon=True).start()
    app.run()
