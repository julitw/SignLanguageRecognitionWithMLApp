from flask import Flask, render_template, Response,  jsonify
import cv2
import numpy as  np
from mediapipe_detection import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz, colors
from model import model, actions
import threading
import numpy as np


app = Flask(__name__)
camera = cv2.VideoCapture(0)
thread = None
sequence = []
sentence = []
predictions = []
threshold = 0.85
res = np.array([0,0,0,0,0,0,0,0,0,0])

def generate_frames():
    global sequence, predictions, sentence, threshold,  res
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera_active:
            success, frame = camera.read()
            if not success:
                break

            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            if (not(results.left_hand_landmarks) and not(results.right_hand_landmarks)):
                sequence = []
            else:
                sequence.append(keypoints)
                sequence = sequence[-30:]
            

            if len(sequence) == 30:
                    print('JUZZZ')
                # Predykcja za pomocą modelu GRU
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print('Res ', res)
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 6:
                        sentence = sentence[6:]

                    # Viz probabilities
                    # image = prob_viz(res, actions, image, colors)
                    print(sentence)

            # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            # # cv2.putText(image, ' '.join(sentence), (3, 30),
            # #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                



            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            

def start_camera_stream():
    global camera_active
    camera_active = True
    global thread
    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=generate_frames)
        thread.start()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'stopped'})

@app.route('/start_camera')
def start_camera():
    start_camera_stream()
    return jsonify({'status': 'start'})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_sentence')
def get_sentence():
    global sentence
    return jsonify({'sentence': ' '.join(sentence)})

@app.route('/clear_sequence')
def clear_sequence():
    global sentence
    sentence = []  # Czyszczenie listy 'sequence'
    return jsonify({'status': 'cleared'})


@app.route('/get_probabilities')
def get_probabilities():
    global res
    
    # Convert ndarray to list
    res_list = res.tolist() if res is not None else []

    return jsonify({'res': res_list})

if __name__ == '__main__':
    app.run(debug=True)
