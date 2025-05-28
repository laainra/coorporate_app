# coorporate_app/app/ai/face_recognition.py
import cv2
import os
import json
import numpy as np
from datetime import datetime, timedelta
from flask import current_app, jsonify # current_app untuk config
from app import db # Import db instance
from app.models import Personnels, Personnel_Images, Camera_Settings, Work_Timer, Personnel_Entries # Import all models
from app.utils.config_variables import get_personnel_folder_path # Import var equivalent
from app.ai.utils import get_camera_instance, process_attendance_entry # Import AI utilities

# ====================================================================
# Global AI/CV Settings & Initialization
# ====================================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths for models and data (derived from app config)
MODEL_LBPH_FILENAME = 'model_lbph.xml'
LABEL_TO_NAME_FILENAME = 'label_to_name.json'

# Global variables for real-time tracking (consider using a more robust state management for production)
# These globals should be managed carefully, especially in multi-threaded/multi-process environments.
# For Flask's development server, they might work, but for production, use Redis or similar.
detection_times = {}
last_detection_time = {}
is_face_detected_global = False
last_save_time_global = datetime.now()

# ====================================================================
# Helper Functions for AI/CV
# ====================================================================

def get_model_path():
    return os.path.join(current_app.config['TRAINED_MODELS_PATH'], MODEL_LBPH_FILENAME)

def get_label_to_name_path():
    return os.path.join(current_app.config['TRAINED_MODELS_PATH'], LABEL_TO_NAME_FILENAME)

def load_label_to_name():
    """Load label-to-name mapping from a JSON file. Create the file if it does not exist."""
    file_path = get_label_to_name_path()
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)
        return {}
    with open(file_path, 'r') as f:
        return json.load(f)

def save_label_to_name(label_to_name):
    """Save label-to-name mapping to a JSON file."""
    file_path = get_label_to_name_path()
    with open(file_path, 'w') as f:
        json.dump(label_to_name, f)

# ====================================================================
# AI/CV Core Logic
# ====================================================================

def capture_faces_logic(personnel):
    """
    Logika inti untuk menangkap gambar wajah dari webcam untuk dataset.
    Ini adalah fungsi inti, bukan view Flask.
    """
    face_id = personnel.id
    face_name = personnel.name
    personnel_folder = os.path.join(get_personnel_folder_path(), face_name)
    os.makedirs(personnel_folder, exist_ok=True)

    cap = get_camera_instance(0) # Menggunakan indeks 0 untuk webcam default
    if not cap or not cap.isOpened():
        return {'status': 'error', 'message': 'Failed to open camera for capture.'}

    count = len(os.listdir(personnel_folder))
    captured_faces = 0

    while captured_faces < 50: # Tangkap hingga 50 wajah baru
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_detected_in_frame:
            face_roi = gray[y:y + h, x:x + w]
            
            # Simpan gambar hanya jika ada wajah terdeteksi
            if face_roi.size > 0: # Pastikan ROI tidak kosong
                count += 1
                file_name = f"face_{face_id}_{face_name}_{count}.jpg"
                file_path_abs = os.path.join(personnel_folder, file_name)
                cv2.imwrite(file_path_abs, face_roi)
                captured_faces += 1

                # Simpan path gambar ke database
                relative_path_to_upload_folder = os.path.relpath(file_path_abs, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
                new_personnel_image = Personnel_Images(
                    personnel_id=personnel.id,
                    image_path=relative_path_to_upload_folder
                )
                db.session.add(new_personnel_image)
                # db.session.commit() # Commit di luar loop untuk performa lebih baik

                if captured_faces >= 50:
                    break
    
    db.session.commit() # Commit semua gambar yang baru ditambahkan
    # cap.release() # Jangan rilis di sini jika get_camera_instance mengelola cache

    if captured_faces > 0:
        return {'status': 'success', 'message': f'Face capture completed. Captured {captured_faces} faces.'}
    else:
        return {'status': 'error', 'message': 'No faces captured. Ensure your face is visible.'}


def _train_face_model_logic():
    """Melatih model pengenalan wajah LBPH dari gambar-gambar di dataset."""
    faces = []
    labels = []
    target_size = (200, 200)
    label_to_name_map = {}

    personnel_base_folder = get_personnel_folder_path()
    
    if not os.path.exists(personnel_base_folder):
        return {'status': 'error', 'message': 'Personnel base folder does not exist.', 'success': False}

    all_personnels = Personnels.query.all()
    personnel_id_to_name = {p.id: p.name for p in all_personnels}

    for personnel_obj in all_personnels:
        name_folder = personnel_obj.name # Nama folder adalah nama personel
        name_folder_path = os.path.join(personnel_base_folder, name_folder)
        
        if not os.path.isdir(name_folder_path):
            print(f"Warning: Personnel folder '{name_folder_path}' not found. Skipping.")
            continue

        label_to_name_map[personnel_obj.id] = personnel_obj.name # Map ID to name

        for file_name in os.listdir(name_folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(name_folder_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                
                # Deteksi wajah di gambar sebelum menambahkannya ke dataset
                faces_in_img = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                if len(faces_in_img) > 0:
                    (x, y, w, h) = faces_in_img[0] # Ambil wajah pertama
                    face_roi = img[y:y+h, x:x+w]
                    face_roi_resized = cv2.resize(face_roi, target_size)
                    
                    faces.append(face_roi_resized)
                    labels.append(np.int32(personnel_obj.id)) # Gunakan ID personel sebagai label
                else:
                    print(f"No face detected in {img_path}, skipping for training.")

    if len(faces) > 0:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(np.array(faces), np.array(labels))
        recognizer.save(get_model_path())
        save_label_to_name(label_to_name_map) # Simpan map ID ke nama
        return {'status': 'success', 'message': 'Model successfully trained and saved.', 'success': True}
    else:
        return {'status': 'error', 'message': 'No face data to train.', 'success': False}


def generate_face_recognition_frames(camera_source, cam_settings=None):
    """
    Generator untuk streaming video frames dengan pengenalan wajah.
    Melakukan deteksi, pengenalan, dan menyimpan data ke DB (Work_Timer/Presence).
    """
    global detection_times, last_detection_time, is_face_detected_global, last_save_time_global
    
    cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
    if not cap or not cap.isOpened():
        print(f"Error: Camera at {camera_source} could not be opened for recognition stream.")
        # Stream a dummy frame with error message
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "CAMERA ERROR!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', dummy_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return # Keluar dari generator

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_file_path = get_model_path()
    if not os.path.exists(model_file_path):
        print("Model not available, please train the model first.")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "MODEL NOT TRAINED!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', dummy_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        cap.release()
        return

    recognizer.read(model_file_path)
    label_to_name = load_label_to_name()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_source}. Attempting to re-open.")
            cap.release()
            cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
            if not cap or not cap.isOpened():
                print(f"Failed to re-open camera {camera_source}. Stopping stream.")
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Reset detection state if no faces are currently detected in frame
        if len(faces_detected_in_frame) == 0:
            is_face_detected_global = False
            # Optional: Anda bisa menambahkan logika untuk mereset timer jika tidak ada wajah
            # yang terdeteksi selama periode tertentu.

        for (x, y, w, h) in faces_detected_in_frame:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)

            name = label_to_name.get(label, "Unknown")
            timer_text = ""
            display_color = (0, 0, 255) # Default Red for Unknown

            # Confidence threshold: Lower value means better match for LBPH (0 is perfect match)
            if confidence < 70: # Sesuaikan threshold ini
                display_color = (0, 255, 0) # Green for Recognized
                
                # Logic for tracking camera role (Work_Timer)
                if cam_settings and cam_settings.role_camera == Camera_Settings.ROLE_TRACKING and name != "Unknown":
                    current_time = datetime.now()
                    if name in last_detection_time:
                        elapsed_time = (current_time - last_detection_time[name]).total_seconds()
                        detection_times[name] += elapsed_time
                    else:
                        detection_times[name] = 0 # Initialize if first detection
                    last_detection_time[name] = current_time
                    is_face_detected_global = True # Set global flag

                    total_time_recognized = int(detection_times.get(name, 0))
                    timer_text = f'Timer: {total_time_recognized}s'

                    # Save to database every minute (or configurable interval)
                    if (current_time - last_save_time_global) >= timedelta(minutes=1):
                        try:
                            personnel_obj = Personnels.query.filter_by(name=name).first()
                            if personnel_obj and cam_settings:
                                new_work_timer = Work_Timer(
                                    personnel_id=personnel_obj.id,
                                    camera_id=cam_settings.id,
                                    type=Work_Timer.TYPE_FACE_DETECTED,
                                    datetime=datetime.utcnow(), # Use UTC for DB storage
                                    timer=int(detection_times[name])
                                )
                                db.session.add(new_work_timer)
                                db.session.commit()
                                print(f"Saved Work_Timer for {name}: {detection_times[name]}s")
                            else:
                                print(f"Skipping Work_Timer save for {name}: Personnel or Camera settings not found.")
                        except Exception as e:
                            db.session.rollback()
                            print(f"Error saving Work_Timer to database: {e}")
                        last_save_time_global = current_time # Update the last save time

                # Logic for presence camera role (Personnel_Entries)
                elif cam_settings and cam_settings.role_camera == Camera_Settings.ROLE_PRESENCE and name != "Unknown":
                    current_time_for_presence = datetime.now()
                    
                    # Define save directory for presence images
                    presence_images_base_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'presence_images')
                    save_directory = os.path.join(presence_images_base_path, current_time_for_presence.strftime('%Y%m%d'))
                    os.makedirs(save_directory, exist_ok=True)
                    
                    image_filename = f"presence_{name}_{current_time_for_presence.strftime('%H%M%S')}.jpg"
                    full_image_path = os.path.join(save_directory, image_filename)
                    
                    # Save the detected face region (ROI)
                    cv2.imwrite(full_image_path, frame[y:y+h, x:x+w])

                    # Get relative path for DB storage
                    relative_path_to_upload_folder = os.path.relpath(full_image_path, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")

                    data_for_attendance = {
                        'name': name,
                        'datetime': current_time_for_presence.strftime('%Y-%m-%d %H:%M:%S'),
                        'image_path': relative_path_to_upload_folder,
                        'camera_id': cam_settings.id
                    }
                    result_status = process_attendance_entry(data_for_attendance)
                    print(f"Attendance processing for {name}: {result_status}")
                    # You might want to flash messages to frontend if this is not a pure stream
            else:
                # Not recognized or low confidence
                name = "Unknown" # Override name if confidence is too low
                display_color = (0, 0, 255) # Red for Unknown
                # If an unknown face is detected, reset its timer if it had one
                if name in last_detection_time:
                    del last_detection_time[name]
                    if name in detection_times:
                        del detection_times[name]
                is_face_detected_global = False

            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), display_color, 2)
            display_text = f'{name} ({confidence:.2f}) {timer_text}'
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)

        # Encode frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # Pastikan kamera dilepas ketika generator berhenti
    release_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)

# ====================================================================
# YOLO Integration (Placeholder)
# ====================================================================
# from ultralytics import YOLO

# _yolo_model = None
# def get_yolo_model():
#     global _yolo_model
#     if _yolo_model is None:
#         # Load your YOLO model here (e.g., from ultralytics)
#         # Ensure the model file (e.g., yolov8n.pt) is in your static/trained_models folder
#         model_path = os.path.join(current_app.config['TRAINED_MODELS_PATH'], 'yolov8n.pt')
#         if os.path.exists(model_path):
#             _yolo_model = YOLO(model_path)
#             print(f"YOLO model loaded from {model_path}")
#         else:
#             print(f"Warning: YOLO model not found at {model_path}. Please download it.")
#             # Fallback to a dummy model or error handling
#             _yolo_model = None
#     return _yolo_model

# def generate_yolo_frames(camera_source, cam_settings=None):
#     """Generator function to stream video frames with YOLO object detection."""
#     model = get_yolo_model()
#     if model is None:
#         dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#         cv2.putText(dummy_frame, "YOLO MODEL NOT FOUND!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         _, jpeg = cv2.imencode('.jpg', dummy_frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#         return

#     cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
#     if not cap or not cap.isOpened():
#         print(f"Error: Camera at {camera_source} could not be opened for YOLO stream.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Failed to read frame from camera {camera_source}. Attempting to re-open.")
#             cap.release()
#             cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
#             if not cap or not cap.isOpened():
#                 print(f"Failed to re-open camera {camera_source}. Stopping stream.")
#                 break
#             continue

#         results = model(frame, stream=True) # Perform inference

#         for r in results:
#             annotated_frame = r.plot() # Plot results on the frame

#             ret, buffer = cv2.imencode('.jpg', annotated_frame)
#             if not ret:
#                 break
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     
#     release_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)