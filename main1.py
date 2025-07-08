# main.py
import numpy as np 
from flask import Flask, render_template 
from flask_socketio import SocketIO

RED = 0xFF0000
GREEN = 0x00FF00
BLUE = 0x0000FF

app = Flask(__name__)
socketio = SocketIO(app)

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
    socketio.run(app, debug=True)
