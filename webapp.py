from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import face_recognition
import pickle
from werkzeug.utils import secure_filename
from untitled11 import LivenessDetector, QuestionBank, ChallengeValidator
import tempfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATASET_PATH = 'dataset'
ENCODINGS_FILE = 'face_encodings.pkl'

# --- Helper functions from main.py ---
def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def generate_encodings_for_user(name):
    known_encodings = []
    known_names = []
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                known_encodings = data['encodings']
                known_names = data['names']
        except:
            pass
    indices_to_remove = [i for i, n in enumerate(known_names) if n == name]
    for i in reversed(indices_to_remove):
        del known_encodings[i]
        del known_names[i]
    user_path = f"{DATASET_PATH}/{name}"
    if os.path.exists(user_path):
        for image_file in os.listdir(user_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = f"{user_path}/{image_file}"
                try:
                    img = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    data = {
        'encodings': known_encodings,
        'names': known_names
    }
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Encodings saved! Total: {len(known_encodings)} encodings for {len(set(known_names))} users")

def regenerate_all_encodings():
    if not os.path.exists(DATASET_PATH):
        print("No dataset found!")
        return
    users = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(f"{DATASET_PATH}/{d}")]
    if not users:
        print("No users found in dataset!")
        return
    print(f"Regenerating encodings for {len(users)} users...")
    all_encodings = []
    all_names = []
    for user in users:
        user_path = f"{DATASET_PATH}/{user}"
        images = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {user}: {len(images)} images")
        for image_file in images:
            image_path = f"{user_path}/{image_file}"
            try:
                img = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    all_encodings.append(encodings[0])
                    all_names.append(user)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    data = {
        'encodings': all_encodings,
        'names': all_names
    }
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Encodings regenerated! Total: {len(all_encodings)} encodings for {len(set(all_names))} users")

def save_user_images(name, images):
    user_path = os.path.join(DATASET_PATH, name)
    os.makedirs(user_path, exist_ok=True)
    for idx, img in enumerate(images):
        img_path = os.path.join(user_path, f"{idx}.jpg")
        cv2.imwrite(img_path, img)

# --- Web routes ---
@app.route('/')
def index():
    return send_from_directory('.', 'webui.html')

@app.route('/login', methods=['POST'])
def login():
    import time
    files = request.files.getlist('frames')
    if not files or len(files) < 5:
        return jsonify({'success': False, 'message': 'Not enough frames received.'})
    images = []
    for file in files:
        in_memory = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)
        images.append(img)
    # Liveness detection: use all challenges from untitled11.py
    liveness_start = time.time()
    liveness_detector = LivenessDetector()
    question_bank = QuestionBank()
    validator = ChallengeValidator()
    total_questions = len(question_bank.questions)
    passed = 0
    liveness_results = []
    # Split images for each challenge (evenly)
    chunk_size = max(1, len(images) // total_questions)
    for i in range(total_questions):
        question = question_bank.get_question(i)
        validator.start_challenge(question, liveness_detector.total_blinks, liveness_detector)
        # Use a chunk of frames for this challenge
        challenge_imgs = images[i*chunk_size:(i+1)*chunk_size]
        if not challenge_imgs:
            challenge_imgs = [images[-1]]
        for img in challenge_imgs:
            liveness_result = liveness_detector.detect_liveness(img)
        result = validator.validate_result(question, liveness_result)
        liveness_results.append({'challenge': question['text'], 'result': result})
        if result == 'pass':
            passed += 1
    liveness_time = time.time() - liveness_start
    print(f"Liveness check took {liveness_time:.2f} seconds.")
    if passed < (total_questions // 2) + 1:
        return jsonify({'success': False, 'message': f'Liveness test failed. Passed {passed}/{total_questions}.', 'liveness': liveness_results})
    # Face recognition
    recog_start = time.time()
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        return jsonify({'success': False, 'message': 'No registered users found.'})
    rgb_frame = cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        return jsonify({'success': False, 'message': 'No face detected in the last frame.', 'liveness': liveness_results})
    recognized_name = None
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        if name != "Unknown":
            recognized_name = name
            break
    recog_time = time.time() - recog_start
    print(f"Recognition took {recog_time:.2f} seconds.")
    if recognized_name:
        return jsonify({'success': True, 'user': recognized_name, 'liveness': liveness_results})
    else:
        return jsonify({'success': False, 'message': 'Face not recognized.', 'liveness': liveness_results})

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    files = request.files.getlist('frames')
    if not name or not files or len(files) < 5:
        return jsonify({'success': False, 'message': 'Name and enough frames required.'})
    images = []
    for file in files:
        in_memory = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)
        images.append(img)
    save_user_images(name, images)
    generate_encodings_for_user(name)
    return jsonify({'success': True})

@app.route('/regenerate', methods=['POST'])
def regenerate():
    regenerate_all_encodings()
    return jsonify({'success': True})

@app.route('/known_users', methods=['GET'])
def known_users():
    _, known_names = load_encodings()
    unique_names = sorted(set(known_names))
    return jsonify({'users': unique_names})

@app.route('/recognize', methods=['POST'])
def recognize_api():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded.'})
    file = request.files['image']
    in_memory = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)
    # Resize image for faster processing
    max_width = 480
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        return jsonify({'success': False, 'message': 'No registered users found.'})
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    recognized_name = None
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        if name != "Unknown":
            recognized_name = name
            break
    if recognized_name:
        return jsonify({'success': True, 'user': recognized_name})
    else:
        return jsonify({'success': False, 'message': 'Face not recognized.'})

if __name__ == '__main__':
    app.run(debug=True) 