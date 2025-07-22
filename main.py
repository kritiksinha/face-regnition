import cv2
import os
import face_recognition
import numpy as np
import pickle

DATASET_PATH = 'dataset'
ENCODINGS_FILE = 'face_encodings.pkl'

def capture_face(name):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return False
    
    count = 0
    os.makedirs(f"{DATASET_PATH}/{name}", exist_ok=True)
    print(f"Capturing face data for {name}...")
    print("Position your face in the frame and press SPACE to capture, Q to quit")
    
    while count < 10:  # Reduced to 10 for faster processing
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"Captured: {count}/10", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, Q to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Capturing Face Data", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if face_locations:
                cv2.imwrite(f"{DATASET_PATH}/{name}/{count}.jpg", frame)
                count += 1
                print(f"Captured image {count}/10")
            else:
                print("No face detected! Please position your face in the frame.")
        elif key == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    if count > 0:
        print(f"Successfully captured {count} images!")
        # Generate encodings for the new user immediately
        generate_encodings_for_user(name)
        return True
    else:
        print("No images captured!")
        return False

def generate_encodings_for_user(name):
    """Generate and save encodings for a specific user"""
    print(f"Generating encodings for {name}...")
    
    # Load existing encodings if they exist
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
    
    # Remove old encodings for this user (if updating)
    indices_to_remove = [i for i, n in enumerate(known_names) if n == name]
    for i in reversed(indices_to_remove):
        del known_encodings[i]
        del known_names[i]
    
    # Generate new encodings for this user
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
    
    # Save updated encodings
    data = {
        'encodings': known_encodings,
        'names': known_names
    }
    
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Encodings saved! Total: {len(known_encodings)} encodings for {len(set(known_names))} users")

def load_encodings():
    """Load pre-computed encodings from file"""
    if not os.path.exists(ENCODINGS_FILE):
        print("No encodings file found! Please register users first.")
        return [], []
    
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def recognize_face():
    """Recognize faces using pre-computed encodings"""
    print("Loading face encodings...")
    known_encodings, known_names = load_encodings()
    
    if not known_encodings:
        print("No registered users found! Please register first.")
        return
    
    print(f"Loaded {len(known_encodings)} encodings for {len(set(known_names))} users")
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Face recognition started. Press Q to quit...")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Process every 3rd frame for better performance
        if hasattr(recognize_face, 'frame_count'):
            recognize_face.frame_count += 1
        else:
            recognize_face.frame_count = 0
        
        if recognize_face.frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Store results for next frames
            recognize_face.last_results = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                confidence = "Low"
                
                if True in matches:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        confidence = f"{(1 - face_distances[best_match_index]) * 100:.1f}%"
                
                recognize_face.last_results.append({
                    'location': (top, right, bottom, left),
                    'name': name,
                    'confidence': confidence
                })
        
        # Draw results (use last computed results for smooth display)
        if hasattr(recognize_face, 'last_results'):
            for result in recognize_face.last_results:
                top, right, bottom, left = result['location']
                name = result['name']
                confidence = result['confidence']
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, f"{name} ({confidence})", (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press Q to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition Login", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

def regenerate_all_encodings():
    """Regenerate encodings for all users in dataset"""
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
    
    # Save all encodings
    data = {
        'encodings': all_encodings,
        'names': all_names
    }
    
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Encodings regenerated! Total: {len(all_encodings)} encodings for {len(set(all_names))} users")

def main():
    print("\n=== Face Recognition Login System ===")
    print("1. Login with Face")
    print("2. Register New User")
    print("3. Regenerate All Encodings")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\nStarting face recognition...")
        recognize_face()
    
    elif choice == '2':
        name = input("Enter new user's name: ").strip()
        if name:
            if capture_face(name):
                print(f"User '{name}' registered successfully!")
            else:
                print("Registration failed!")
        else:
            print("Invalid name!")
    
    elif choice == '3':
        print("\nRegenerating all encodings...")
        regenerate_all_encodings()
        
    elif choice == '4':
        print("Goodbye!")
        
    else:
        print("Invalid choice! Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()