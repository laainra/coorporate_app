# coorporate_app/app/routes/stream_routes.py
import cv2
import os
import json
import numpy as np
from datetime import datetime, timedelta, date, time
from flask import Blueprint, Response, request, jsonify, render_template, current_app, flash, redirect, url_for
from flask_login import login_required, current_user
from app.models import Personnels, Personnel_Images, Camera_Settings, Work_Timer, Personnel_Entries, Company, Divisions # Import semua model
from app import db # Import instance db
from app.utils.decorators import admin_required, employee_required # Decorators
from sqlalchemy.sql import text 
import re
import time
from threading import Thread, Lock
# ====================================================================
# Global AI/CV Settings & Initialization
# ====================================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths for models and data (derived from app config)
MODEL_LBPH_FILENAME = 'model_lbph.xml'
LABEL_TO_NAME_FILENAME = 'label_to_name.json'

# Global dict untuk menyimpan frame
camera_frames = {}
camera_locks = {}



# Global variables for real-time tracking (WARNING: thread-safety in production)
detection_times = {}
last_detection_time = {}
last_save_time_global = datetime.now()
_camera_instance_cache = {} # Cache for cv2.VideoCapture instances

bp = Blueprint('stream', __name__, template_folder='../templates') # Menggunakan template folder global app/templates/

# ====================================================================
# Helper Functions (Path, Load/Save Label Map, Camera Management, etc.)
# ====================================================================

def get_personnel_folder_path():
    # Asumsi: UPLOAD_FOLDER = os.path.join(basedir, 'app', 'static','img')
    # Dan personnel_pics adalah sub-folder dari UPLOAD_FOLDER
    return os.path.join(current_app.config['UPLOAD_FOLDER'], 'personnel_pics')

def get_model_path():
    return os.path.join(current_app.config['TRAINED_MODELS_PATH'], MODEL_LBPH_FILENAME)

def get_model(model_folder_path):
    return os.path.join(model_folder_path, MODEL_LBPH_FILENAME)

def get_label_to_name_path():
    return os.path.join(current_app.config['TRAINED_MODELS_PATH'], LABEL_TO_NAME_FILENAME)

def load_label_to_name():
    file_path = get_label_to_name_path()
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)
        return {}
    with open(file_path, 'r') as f:
        data = json.load(f)  # <-- tambahkan ini
        return {int(k): v for k, v in data.items()}
    
def load_label(model_folder_path):
    file_path = os.path.join(model_folder_path, LABEL_TO_NAME_FILENAME)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)
        return {}
    with open(file_path, 'r') as f:
        data = json.load(f)  # <-- tambahkan ini
        return {int(k): v for k, v in data.items()}
    
def save_label_to_name(label_to_name):
    file_path = get_label_to_name_path()
    with open(file_path, 'w') as f:
        json.dump(label_to_name, f)
        
def get_camera_instance(camera_source):
    global _camera_instance_cache
    processed_source = camera_source
    if isinstance(camera_source, str) and camera_source.isdigit():
        processed_source = int(camera_source)

    # Check cache
    if processed_source in _camera_instance_cache:
        cap = _camera_instance_cache[processed_source]
        if cap.isOpened():
            return cap
        else:
            print(f"[get_camera_instance] Cached camera for {processed_source} is not opened. Releasing.")
            cap.release()
            del _camera_instance_cache[processed_source]

    cap = None
    if isinstance(processed_source, int):
        if os.name == 'nt':
            cap = cv2.VideoCapture(processed_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(processed_source, cv2.CAP_MSMF)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(processed_source, cv2.CAP_V4L2)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(processed_source)
    else:
        cap = cv2.VideoCapture(processed_source, cv2.CAP_FFMPEG)

    if cap and cap.isOpened():
        _camera_instance_cache[processed_source] = cap
        print(f"Camera instance for source '{processed_source}' opened and cached.")
    else:
        print(f"Error: Could not open camera source '{processed_source}'.")
        return None

    return cap

def release_camera_instance(camera_source):
    if camera_source in _camera_instance_cache:
        cap = _camera_instance_cache[camera_source]
        if cap.isOpened():
            cap.release()
        del _camera_instance_cache[camera_source]
        print(f"Camera instance for source '{camera_source}' released.")




def get_today_presences_logic(company_id):
    today_str = date.today().strftime('%Y-%m-%d') # Format tanggal sebagai string untuk query

    # Query diadaptasi dari contoh Django Anda, dengan penyesuaian untuk Flask/SQLAlchemy
    # dan penambahan filter company_id.
    # CATATAN: TIMESTAMPDIFF dan CONCAT adalah sintaks MySQL.
    # Sesuaikan jika Anda menggunakan database lain.
    raw_query = text(f"""
        SELECT 
            p.id AS personnel_id,
            p.name AS personnel_name,
            MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END) AS attended_time,
            MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END) AS leaving_time,
            CASE 
                WHEN EXISTS (
                    SELECT 1 
                    FROM personnel_entries AS sub_pe
                    WHERE sub_pe.personnel_id = p.id 
                    AND DATE(sub_pe.timestamp) = :today_date
                    AND sub_pe.presence_status = 'LEAVE'
                ) THEN 'LEAVING' /* Jika ada record LEAVE, status terakhir dianggap LEAVING */
                ELSE MAX(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.presence_status END)
            END AS latest_status,
            TIMESTAMPDIFF(HOUR, 
                MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END),
                MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END)
            ) AS work_hours,
            CASE 
                WHEN TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END),
                    MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END)
                ) > 8 THEN CONCAT('Overtime ', TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END),
                    MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END)
                ) - 8, ' hours')
                WHEN TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END),
                    MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END)
                ) < 8 AND MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END) IS NOT NULL THEN CONCAT('Less time ', 8 - TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END),
                    MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END)
                ), ' hours')
                WHEN MAX(CASE WHEN pe.presence_status = 'LEAVE' THEN pe.timestamp END) IS NULL AND MIN(CASE WHEN pe.presence_status IN ('ONTIME', 'LATE') THEN pe.timestamp END) IS NOT NULL THEN 'Still Working'
                ELSE 'Standard Time'
            END AS notes,
            (SELECT sub_img.image
             FROM personnel_entries AS sub_img
             WHERE sub_img.personnel_id = p.id 
             AND DATE(sub_img.timestamp) = :today_date
             AND sub_img.presence_status IN ('ONTIME', 'LATE')
             ORDER BY sub_img.timestamp DESC 
             LIMIT 1
            ) AS attendance_image_path,
            (SELECT sub_img_leave.image /* Ganti 'image' dengan nama kolom path gambar Anda */
             FROM personnel_entries AS sub_img_leave
             WHERE sub_img_leave.personnel_id = p.id 
             AND DATE(sub_img_leave.timestamp) = :today_date
             AND sub_img_leave.presence_status = 'LEAVE'
             ORDER BY sub_img_leave.timestamp DESC 
             LIMIT 1
            ) AS leaving_image_path
        FROM 
            personnel_entries AS pe
        JOIN 
            personnels AS p ON p.id = pe.personnel_id
        WHERE 
            DATE(pe.timestamp) = :today_date
            AND p.company_id = :company_id  /* Filter berdasarkan company_id */
        GROUP BY p.id, p.name /* Grup berdasarkan ID dan nama personel */
        ORDER BY p.name; 
    """) # Tambahkan ORDER BY jika diperlukan

    try:
        result = db.session.execute(raw_query, {"today_date": today_str, "company_id": company_id.id})
        entries = result.mappings().all() # Mengambil hasil sebagai list of dictionaries

        formatted_presences = []
        for entry in entries:
            # Sesuaikan dengan output yang dibutuhkan frontend: name, status, timestamp
            # 'status' diambil dari 'latest_status'
            # 'timestamp' bisa diambil dari 'attended_time' (waktu masuk pertama)
            
            attended_time_val = entry.get('attended_time')
            timestamp_to_display = None
            if isinstance(attended_time_val, datetime):
                timestamp_to_display = attended_time_val.strftime('%H:%M:%S')
            elif isinstance(attended_time_val, str): # Jika DB mengembalikan string
                 try:
                    timestamp_to_display = datetime.fromisoformat(attended_time_val).strftime('%H:%M:%S')
                 except ValueError:
                    timestamp_to_display = attended_time_val # Biarkan apa adanya jika tidak bisa diparse
            
            # Debugging:
            # current_app.logger.debug(f"Entry: {entry}")
            # current_app.logger.debug(f"Attended time type: {type(attended_time_val)}, value: {attended_time_val}")
            # current_app.logger.debug(f"Timestamp to display: {timestamp_to_display}")

            formatted_presences.append({
                'name': entry.get('personnel_name'),
                'status': entry.get('latest_status', 'N/A'), # Default ke 'N/A' jika tidak ada
                'timestamp': timestamp_to_display if timestamp_to_display else '-', # Waktu masuk pertama
                # Anda bisa menambahkan field lain jika frontend diubah untuk menampilkannya:
                # 'personnel_id': entry.get('personnel_id'),
                # 'attended_time': timestamp_to_display if timestamp_to_display else '-',
                # 'leaving_time': entry.get('leaving_time').strftime('%H:%M:%S') if entry.get('leaving_time') else '-',
                # 'work_hours': entry.get('work_hours'),
                # 'notes': entry.get('notes'),
                # 'attendance_image_path': entry.get('attendance_image_path'),
                # 'leaving_image_path': entry.get('leaving_image_path'),
            })
        
        # print(f"Data absensi (raw query) untuk company_id {company_id}: {formatted_presences}")
        return formatted_presences

    except Exception as e:
        current_app.logger.error(f"Error executing raw query in get_today_presences_logic for company_id {company_id}: {e}", exc_info=True)
        return [] # Kembalikan list kosong jika terjadi error


# ====================================================================
# Logika Pengambilan Gambar Wajah (untuk dataset)
# ====================================================================
def capture_faces_logic(personnel):
    face_id = personnel.id
    face_name = personnel.name
    personnel_folder = os.path.join(get_personnel_folder_path(), face_name)
    os.makedirs(personnel_folder, exist_ok=True)

    cap = get_camera_instance(0) # Menggunakan indeks 0 untuk webcam default
    if not cap or not cap.isOpened():
        return {'status': 'error', 'message': 'Failed to open camera for capture.'}

    count = len(os.listdir(personnel_folder))
    captured_faces = 0

    while captured_faces < 50:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_detected_in_frame:
            face_roi = gray[y:y + h, x:x + w]
            
            if face_roi.size > 0:
                count += 1
                file_name = f"face_{face_id}_{face_name}_{count}.jpg"
                file_path_abs = os.path.join(personnel_folder, file_name)
                cv2.imwrite(file_path_abs, face_roi)
                captured_faces += 1

                relative_path_to_upload_folder = os.path.relpath(file_path_abs, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
                new_personnel_image = Personnel_Images(
                    personnel_id=personnel.id,
                    image_path=relative_path_to_upload_folder
                )
                db.session.add(new_personnel_image)

                if captured_faces >= 50:
                    break
    
    db.session.commit()
    
    if captured_faces > 0:
        return {'status': 'success', 'message': f'Face capture completed. Captured {captured_faces} faces.'}
    else:
        return {'status': 'error', 'message': 'No faces captured. Ensure your face is visible.'}

# ====================================================================
# Logika Pelatihan Model
# ====================================================================
def _train_face_model_logic():
    faces = []
    labels = []
    target_size = (200, 200)
    label_to_name_map = {}

    personnel_base_folder = get_personnel_folder_path()
    
    if not os.path.exists(personnel_base_folder):
        return {'status': 'error', 'message': 'Personnel base folder does not exist.', 'success': False}

    all_personnels = Personnels.query.all()

    for personnel_obj in all_personnels:
        name_folder = personnel_obj.name
        name_folder_path = os.path.join(personnel_base_folder, name_folder)
        
        if not os.path.isdir(name_folder_path):
            print(f"Warning: Personnel folder '{name_folder_path}' not found. Skipping.")
            continue

        label_to_name_map[personnel_obj.id] = personnel_obj.name

        for file_name in os.listdir(name_folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(name_folder_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                
                faces_in_img = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                if len(faces_in_img) > 0:
                    (x, y, w, h) = faces_in_img[0]
                    face_roi = img[y:y+h, x:x+w]
                    face_roi_resized = cv2.resize(face_roi, target_size)
                    
                    faces.append(face_roi_resized)
                    labels.append(np.int32(personnel_obj.id))
                else:
                    pass

    if len(faces) > 0:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(np.array(faces), np.array(labels))
        recognizer.save(get_model_path())
        save_label_to_name(label_to_name_map)
        return {'status': 'success', 'message': 'Model successfully trained and saved.', 'success': True}
    else:
        return {'status': 'error', 'message': 'No face data to train.', 'success': False}


# ====================================================================
# Flask Routes for Stream Blueprint (All Consolidated)
# ====================================================================

@bp.route('/video_feed/<cam_source>')
@login_required

def video_feed(cam_source):
    """Route for raw camera feed (no AI processing)."""
    try:
        camera_source_parsed = int(cam_source) if cam_source.isdigit() else cam_source
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid camera source format.'}), 400

    return Response(generate_raw_video_frames(camera_source_parsed),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




# ====================================================================
# Routes untuk Fungsionalitas Face Recognition (dari face_rec.py Django)
# Dipindahkan ke dalam blueprint 'stream'
# ====================================================================

@bp.route('/capture')
@login_required
@employee_required
def capture_page():
    """Renders the page for capturing new face images for a personnel."""
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Personnel data not found for your user account. Please contact admin.", "danger")
        return redirect(url_for('employee.dashboard')) 
    
    return render_template('employee_panel/capture.html', name=personnel.name, personnel_id=personnel.id)

@bp.route('/capture_data', methods=['POST'], endpoint='capture_faces')
@login_required

def capture_faces():
    """Endpoint to capture face images from webcam and save them for personnel dataset."""
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        return jsonify({'status': 'error', 'message': 'User personnel data not found.'}), 400

    result = capture_faces_logic(personnel) # Call the core logic
    
    if result['status'] == 'success':
        flash(result['message'], 'success')
    else:
        flash(result['message'], 'danger')
        
    return jsonify(result)

# Variabel global tracking waktu (optional)
last_detection_time = {}
detection_times = {}

@bp.route('/capture_video')
@login_required
def capture_video():
    # Ambil model path di sini, dalam konteks aplikasi yang benar
    model_file_path = get_model_path()
    
    label_to_name = load_label_to_name()

    def generate_frames():
        if not os.path.exists(model_file_path):
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "MODEL NOT TRAINED!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', dummy_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file_path)

        

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

                label, confidence = recognizer.predict(face_roi)
                name = label_to_name.get(label, "Unknown")

                if confidence < 70:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    name = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@bp.route('/train', methods=['POST'])
@login_required
def train_model():
    result = _train_face_model_logic()
    
    # Deteksi jika request dari JavaScript/AJAX (bukan browser)
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(result)  # Kirim JSON jika dari fetch()
    
    # Kalau dari browser biasa, pakai redirect + flash
    if result['success']:
        flash(result['message'], 'success')
    else:
        flash(result['message'], 'danger')
    return redirect(url_for('stream.dataset_no_id'))

# ====================================================================
# Logika Capture Video (untuk dataset)
# ====================================================================
def generate_raw_video_frames(camera_source):
    cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
    if not cap or not cap.isOpened():
        print(f"Error: Camera at {camera_source} could not be opened for raw stream.")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "CAMERA ERROR!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', dummy_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_source}. Reattempting to open.")
            cap.release()
            cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
            if not cap or not cap.isOpened():
                print(f"Failed to reopen camera {camera_source}. Stopping stream.")
                break
            continue
        
        _, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    release_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
# ====================================================================
# Logika Streaming dengan Pengenalan Wajah (Core Generator)
# ====================================================================
def generate_face_recognition_frames(camera_source, cam_settings, model_folder_path, app, db):
    global detection_times, last_detection_time, last_save_time_global
    
    cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
    if not cap or not cap.isOpened():
        print(f"Error: Camera at {camera_source} could not be opened for recognition stream.")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "CAMERA ERROR!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', dummy_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_file_path = get_model(model_folder_path)
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
    label_to_name = load_label(model_folder_path) if model_folder_path else load_label_to_name()

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
        
        if len(faces_detected_in_frame) == 0:
            pass

        for (x, y, w, h) in faces_detected_in_frame:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_roi)

            name = label_to_name.get(label, "Unknown")
            timer_text = ""
            display_color = (0, 0, 255)

            if confidence < 70:
                display_color = (0, 255, 0)
                
                if cam_settings and cam_settings.role_camera == Camera_Settings.ROLE_TRACKING and name != "Unknown":
                    current_time = datetime.now()
                    if name in last_detection_time:
                        elapsed_time = (current_time - last_detection_time[name]).total_seconds()
                        detection_times[name] += elapsed_time
                    else:
                        detection_times[name] = 0
                    last_detection_time[name] = current_time
                    
                    total_time_recognized = int(detection_times.get(name, 0))
                    timer_text = f'Timer: {total_time_recognized}s'

                    if (current_time - last_save_time_global) >= timedelta(minutes=1):
                        try:
                            with app.app_context():
                                personnel_obj = db.session.query(Personnels).filter_by(name=name).first()
                                if personnel_obj and cam_settings:
                                    new_work_timer = Work_Timer(
                                        personnel_id=personnel_obj.id,
                                        camera_id=cam_settings.id,
                                        type=Work_Timer.TYPE_FACE_DETECTED,
                                        datetime=datetime.utcnow(),
                                        timer=int(detection_times[name])
                                    )
                                    db.session.add(new_work_timer)
                                    db.session.commit()
                                    print(f"Saved Work_Timer for {name}: {detection_times[name]}s")
                                else:
                                    print(f"Skipping Work_Timer save for {name}: Personnel or Camera settings not found.")
                        except Exception as e:
                            with app.app_context():
                                db.session.rollback()
                            print(f"Error saving Work_Timer to database: {e}")


                        last_save_time_global = current_time
            else:
                name = "Unknown"
                display_color = (0, 0, 255)
                if name in last_detection_time:
                    del last_detection_time[name]
                    if name in detection_times:
                        del detection_times[name]

            cv2.rectangle(frame, (x, y), (x+w, y+h), display_color, 2)
            cv2.putText(frame, f'{name} ({confidence:.2f}), {timer_text}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    release_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)

@bp.route('/recognize/<int:cam_id>')
@login_required
# Atau employee_required jika ini untuk melihat stream dengan pengenalan
def predict_video(cam_id):
    """Route for streaming video with object detection (face recognition)."""
    try:
        if cam_id == 0 or cam_id == 1: # Handle direct webcam index
            cam_settings = None # No specific DB settings for direct webcam
            camera_source_url = cam_id
        else:
            cam_settings = Camera_Settings.query.filter_by(id=cam_id, cam_is_active=True).first()
            if not cam_settings:
                return jsonify({'status': 'error', 'message': 'Camera not found or inactive.'}), 404
            camera_source_url = cam_settings.feed_src
            
        # Ensure only authorized user can view streams for their company
        if current_user.is_admin and cam_settings and cam_settings.company_obj != current_user.company_linked:
             return jsonify({'status': 'error', 'message': 'Unauthorized camera access.'}), 403
        if current_user.is_employee: # Employees can't view arbitrary streams (unless it's their own data capture)
             return jsonify({'status': 'error', 'message': 'Unauthorized stream access for employee role.'}), 403

                # Ambil path model dari config dalam konteks aktif
        model_folder_path = current_app.config['TRAINED_MODELS_PATH']

        app_context = current_app._get_current_object()

        return Response(
            generate_face_recognition_frames(camera_source_url, cam_settings, model_folder_path, app_context, db),
            mimetype='multipart/x-mixed-replace; boundary=frame'
)
        
    except Exception as e:
        print(f"Error in predict_video_stream: {e}")
        return jsonify({'status': 'error', 'message': f'Internal server error: {str(e)}'}), 500


@bp.route('/dataset')
@bp.route('/dataset/<int:personnel_id>')
@login_required
# Admin can see all, employee can only see their own
def dataset_no_id(personnel_id=None):
    """Displays the dataset of face images for a given personnel."""
    current_personnel = None
    if personnel_id:
        current_personnel = Personnels.query.get_or_404(personnel_id)
        # Izinkan superadmin akses semua, admin hanya company sendiri
        if current_user.role == 'admin' and current_personnel.company_obj != current_user.company_linked:
            flash("Unauthorized access to this personnel's dataset.", "danger")
            return redirect(url_for('admin.dashboard')) 
        if current_user.role == 'employee':
            flash("Unauthorized: Employees can only access their own dataset.", "danger")
            return redirect(url_for('employee.dashboard'))
    else:
        current_personnel = Personnels.query.filter_by(user_account=current_user).first()
        if not current_personnel:
            flash("Personnel data not found for your account.", "danger")
            return redirect(url_for('employee.dashboard'))


    images = []
    personnel_folder = os.path.join(get_personnel_folder_path(), current_personnel.name)
    if os.path.exists(personnel_folder):
        for file_name in os.listdir(personnel_folder):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                # URL untuk gambar statis
               relative_path_for_static = '/'.join(['img', os.path.relpath(os.path.join(personnel_folder, file_name), current_app.config['UPLOAD_FOLDER']).replace('\\', '/')])

               images.append({
                    'url': url_for('static', filename=relative_path_for_static),
                    'filename': file_name,
                    'personnel_name': current_personnel.name 
                })
    
    return render_template('employee_panel/dataset.html', images=images, name=current_personnel.name, personnel=current_personnel)

@bp.route('/delete_images', methods=['POST'])
@login_required

def delete_images():
    if request.method == 'POST':
        images_to_delete = request.form.getlist('images_to_delete') # Get list of filenames
        personnel_name = request.form.get('personnel_name') # Get personnel name from form

        personnel = Personnels.query.filter_by(name=personnel_name).first()
        if not personnel:
            flash("Personnel not found.", "danger")
            return redirect(request.referrer or url_for('admin.dashboard'))

        if current_user.is_admin and personnel.company_obj != current_user.company_linked:
            flash("Unauthorized: Personnel not in your company.", "danger")
            return redirect(request.referrer or url_for('admin.dashboard'))
        if current_user.is_employee and personnel.user_account != current_user:
            flash("Unauthorized: You can only delete your own images.", "danger")
            return redirect(request.referrer or url_for('employee.dashboard'))

        deleted_count = 0
        for filename in images_to_delete:
            full_path = os.path.join(get_personnel_folder_path(), personnel.name, filename)
            
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    relative_path_in_db = os.path.relpath(full_path, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
                    img_db_entry = Personnel_Images.query.filter_by(image_path=relative_path_in_db).first()
                    if img_db_entry:
                        db.session.delete(img_db_entry)
                        db.session.commit()
                    deleted_count += 1
                except Exception as e:
                    flash(f"Error deleting {filename}: {str(e)}", "warning")
                    db.session.rollback()

        flash(f"Deleted {deleted_count} selected images.", "success")
        return redirect(request.referrer or url_for('stream.dataset_no_id', personnel_id=personnel.id))
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@bp.route('/get-today-presences') # Anda bisa memilih URL path yang lebih deskriptif
@login_required # Asumsi hanya pengguna yang login bisa mengakses
# @admin_required # Tambahkan jika hanya admin yang boleh
def get_today_presences(): # Nama fungsi ini akan menjadi bagian dari endpoint

    company_id = Company.query.filter_by(user_id=current_user.id).first() 
    try:
        presence_data = get_today_presences_logic(company_id)
        print(f"Presence data for company {company_id.id}: {presence_data}") # Debugging log
        return jsonify({'status': 'success', 'data': presence_data})
    except Exception as e:
        current_app.logger.error(f"Error in get_today_presences for company {company_id}: {str(e)}")
        # Untuk produksi, Anda mungkin tidak ingin mengirim detail error ke klien
        return jsonify({'status': 'error', 'message': 'Could not retrieve presence data due to a server error.'}), 500

def process_attendance_entry(data):
    name = data.get('name')
    datetime_str = data.get('datetime')
    image_path = data.get('image_path') # Ini akan digunakan untuk INSERT
    cam_id = data.get('camera_id')
    
    if not all([name, datetime_str, image_path, cam_id]):
        current_app.logger.error("Data input tidak lengkap untuk process_attendance_entry.")
        return 'input_error' # Atau status error yang sesuai

    try:
        detected_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        detected_date_str = detected_time.strftime('%Y-%m-%d') # Untuk query DATE()
    except ValueError:
        current_app.logger.error(f"Format datetime salah: {datetime_str}")
        return 'input_error'

    personnel_id = None
    # 1. Dapatkan Personnel ID menggunakan raw SQL
    try:
        query_personnel = text("SELECT id FROM personnels WHERE name = :personnel_name LIMIT 1")
        result_personnel = db.session.execute(query_personnel, {"personnel_name": name}).fetchone()
        if result_personnel:
            personnel_id = result_personnel[0]
        else:
            current_app.logger.warning(f"Personel '{name}' tidak ditemukan.")
            return 'personnel_not_found'
    except Exception as e:
        current_app.logger.error(f"Error saat mengambil data personel '{name}': {e}", exc_info=True)
        return 'db_error'
    
    # 1.5. Validasi personel dan kamera berasal dari perusahaan yang sama
    query_company_check = text("""
        SELECT 
            p.company_id AS person_company_id,
            c.company_id AS camera_company_id
        FROM personnels p
        JOIN camera_settings c ON c.id = :cam_id
        WHERE p.id = :p_id
    """)

    try:
        result_company = db.session.execute(query_company_check, {"p_id": personnel_id, "cam_id": cam_id}).fetchone()
        if not result_company:
            current_app.logger.warning(f"Gagal mencocokkan company_id personel dan kamera.")
            return 'mismatch_or_invalid'
        
        if result_company.person_company_id is None or result_company.camera_company_id is None:
            current_app.logger.warning(f"company_id kosong: personel({result_company.person_company_id}), kamera({result_company.camera_company_id})")
            return 'missing_company_id'
        
        if result_company.person_company_id != result_company.camera_company_id:
            current_app.logger.info(f"Personel '{name}' tidak berasal dari company yang sama dengan kamera {cam_id}. Absensi diabaikan.")
            return 'unauthorized_entry'
    except Exception as e:
        current_app.logger.error(f"Error saat validasi company_id personel dan kamera: {e}", exc_info=True)
        return 'db_error'


    # 2. Dapatkan Pengaturan Kamera dan Jumlah Absensi yang Sudah Ada (Raw SQL)
    # Asumsi Camera_Settings.ROLE_PRESENCE adalah 'P'
    role_presence_value = 'P' # Ganti jika nilai konstanta Anda berbeda

    query_settings_and_counts = text(f"""
        SELECT 
            cs.attendance_time_start, 
            cs.attendance_time_end, 
            cs.leaving_time_start, 
            cs.leaving_time_end,
            cs.cam_is_active, 
            (SELECT COUNT(*) FROM personnel_entries pe 
             WHERE pe.personnel_id = :p_id AND DATE(pe.timestamp) = :d_date AND pe.presence_status = 'ONTIME') AS count_ontime,
            (SELECT COUNT(*) FROM personnel_entries pe 
             WHERE pe.personnel_id = :p_id AND DATE(pe.timestamp) = :d_date AND pe.presence_status = 'LATE') AS count_late,
            (SELECT COUNT(*) FROM personnel_entries pe 
             WHERE pe.personnel_id = :p_id AND DATE(pe.timestamp) = :d_date AND pe.presence_status = 'LEAVE') AS count_leave
        FROM camera_settings cs 
        WHERE cs.id = :c_id AND cs.role_camera = :role_cam
    """)
    
    settings_result = None
    try:
        settings_result = db.session.execute(
            query_settings_and_counts,
            {"p_id": personnel_id, "d_date": detected_date_str, "c_id": cam_id, "role_cam": role_presence_value}
        ).fetchone()
    except Exception as e:
        current_app.logger.error(f"Error saat mengambil pengaturan kamera & count absensi untuk cam_id {cam_id}: {e}", exc_info=True)
        return 'db_error'

    if not settings_result:
        current_app.logger.warning(f"Pengaturan kamera tidak ditemukan atau bukan kamera Absensi untuk ID {cam_id}.")
        return 'invalid_camera'
    
    if not settings_result.cam_is_active: # Kolom cam_is_active dari query
        current_app.logger.warning(f"Kamera ID {cam_id} tidak aktif.")
        return 'invalid_camera' # Atau status error spesifik

    # 3. Proses Hasil Query Pengaturan & Count
    # Akses berdasarkan nama kolom jika fetchone() mengembalikan objek Row atau MappingResult
    attendance_start_str = settings_result.attendance_time_start
    attendance_end_str = settings_result.attendance_time_end
    leaving_start_str = settings_result.leaving_time_start
    leaving_end_str = settings_result.leaving_time_end

    has_ontime = settings_result.count_ontime > 0
    has_late = settings_result.count_late > 0
    has_leave = settings_result.count_leave > 0
    
    # 4. Parse Waktu Pengaturan
    try:
        attendance_start = datetime.strptime(attendance_start_str, '%H:%M:%S').time() if attendance_start_str else None
        attendance_end = datetime.strptime(attendance_end_str, '%H:%M:%S').time() if attendance_end_str else None
        leaving_start = datetime.strptime(leaving_start_str, '%H:%M:%S').time() if leaving_start_str else None
        leaving_end = datetime.strptime(leaving_end_str, '%H:%M:%S').time() if leaving_end_str else None
    except ValueError as ve:
        current_app.logger.error(f"Format waktu salah di pengaturan kamera ID {cam_id}: {ve}")
        return 'invalid_camera_settings' # Error baru untuk format waktu salah

    current_time_only = detected_time.time()

    # 5. Tentukan Status (Logika Python dari fungsi asli Anda)
    status = 'UNKNOWN' # Default ke UNKNOWN
    if attendance_start and attendance_end and attendance_start <= current_time_only <= attendance_end:
        status = 'ONTIME'
    elif leaving_start and leaving_end and leaving_start <= current_time_only <= leaving_end:
        status = 'LEAVE'
    else: # Di luar jam ONTIME dan LEAVE
        # Cek apakah di antara akhir jam ONTIME dan awal jam LEAVE (berarti LATE)
        if attendance_end and leaving_start and attendance_end < current_time_only < leaving_start:
            status = 'LATE'
        # Jika tidak, status tetap UNKNOWN (misalnya, datang terlalu awal atau pulang terlalu larut di luar semua window)

    # 6. Logika Bisnis untuk Duplikat/Status (dari fungsi asli Anda)
    if status == 'ONTIME':
        if has_ontime or has_late: # Jika sudah ONTIME atau LATE, tidak bisa ONTIME lagi
            print(f"Duplikat ONTIME/LATE untuk {name}. ONTIME tidak dicatat.")
            return 'already_present' 
    elif status == 'LATE':
        if has_ontime or has_late: # Jika sudah ONTIME atau LATE, tidak bisa LATE lagi
            print(f"Duplikat ONTIME/LATE untuk {name}. LATE tidak dicatat.")
            return 'already_present'
    elif status == 'LEAVE':
        if has_leave:
            print(f"Duplikat LEAVE untuk {name}. LEAVE tidak dicatat.")
            return 'already_present'
        if not (has_ontime or has_late):
            print(f"Tidak bisa LEAVE untuk {name} sebelum ONTIME atau LATE.")
            return 'not_eligible_for_leave'
    elif status == 'UNKNOWN':
        print(f"Status UNKNOWN untuk {name} pada {detected_time}. Tidak ada entri yang dibuat.")
        return 'ignored_unknown_status' # Status khusus untuk diabaikan



    # 7. Simpan Entri Baru (Raw SQL INSERT)
    query_insert = text("""
        INSERT INTO personnel_entries (camera_id, personnel_id, timestamp, presence_status, image)
        VALUES (:cam_id, :p_id, :ts, :status, :img_path)
    """)
    try:
        db.session.execute(query_insert, {
            "cam_id": cam_id,
            "p_id": personnel_id,
            "ts": detected_time,
            "status": status,
            "img_path": image_path # Menggunakan image_path sesuai definisi Anda
        })
        db.session.commit()
        print(f"Berhasil memasukkan entri {status} untuk {name} (ID: {personnel_id}) pada {detected_time} dari kamera {cam_id}")
        return 'success'
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Gagal memasukkan data absensi ke database untuk {name}: {e}", exc_info=True)
        return 'db_error'
  
def start_camera_thread(cam_id, feed_src):
    if cam_id in camera_frames and camera_frames.get(cam_id) is not None:
        print(f"[Thread Manager] Camera {cam_id} appears to be already running or has a frame.")
        return

    old_cap = _camera_instance_cache.pop(cam_id, None)
    if old_cap:
        print(f"[Thread Manager] Releasing previously cached camera instance for {cam_id}.")
        old_cap.release()

    lock = Lock()
    camera_locks[cam_id] = lock
    camera_frames[cam_id] = None

    print(f"[Thread Manager] Starting camera thread for {cam_id} with source: {feed_src}")

    def capture():
        cap = None
        fail_count_read = 0
        fail_count_open = 0
        MAX_OPEN_FAILS_BEFORE_LONG_PAUSE = 5
        MAX_READ_FAILS_BEFORE_REOPEN = 15
        FPS_TARGET = 15

        while True:
            try:
                if not cap or not cap.isOpened():
                    if cap:
                        cap.release()
                        cap = None

                    print(f"[Thread {cam_id}] Attempting to (re)connect to {feed_src}...")
                    cap = get_camera_instance(feed_src)

                    if not cap or not cap.isOpened():
                        fail_count_open += 1
                        with lock:
                            camera_frames[cam_id] = None
                        print(f"[Thread {cam_id}] Failed to open camera ({fail_count_open}). Waiting...")
                        sleep_duration_open = min(fail_count_open * 2, 60)
                        if fail_count_open > MAX_OPEN_FAILS_BEFORE_LONG_PAUSE:
                            sleep_duration_open = 120
                        time.sleep(sleep_duration_open)
                        continue
                    else:
                        print(f"[Thread {cam_id}] Successfully (re)connected to {feed_src}.")
                        _camera_instance_cache[cam_id] = cap
                        fail_count_open = 0
                        fail_count_read = 0

                ret, frame = cap.read()
                if not ret or frame is None:
                    fail_count_read += 1
                    print(f"[Thread {cam_id}] Failed to read frame ({fail_count_read}/{MAX_READ_FAILS_BEFORE_REOPEN}).")
                    with lock:
                        camera_frames[cam_id] = None

                    if fail_count_read >= MAX_READ_FAILS_BEFORE_REOPEN:
                        print(f"[Thread {cam_id}] Max read failures reached. Reinitializing camera.")
                        cap.release()
                        cap = None
                        _camera_instance_cache.pop(cam_id, None)
                        fail_count_read = 0
                        time.sleep(2)
                    else:
                        time.sleep(0.1)
                    continue
                else:
                    fail_count_read = 0
                    with lock:
                        camera_frames[cam_id] = frame.copy()
                    time.sleep(1 / FPS_TARGET)

            except cv2.error as e:
                print(f"[Thread {cam_id}] OpenCV Error: {e}. Args: {e.args}")
                with lock:
                    camera_frames[cam_id] = None
                if cap:
                    cap.release()
                cap = None
                _camera_instance_cache.pop(cam_id, None)
                fail_count_read = 0
                fail_count_open = 0
                time.sleep(5)

            except Exception as e:
                print(f"[Thread {cam_id}] Unexpected error: {e}")
                with lock:
                    camera_frames[cam_id] = None
                if cap:
                    cap.release()
                cap = None
                _camera_instance_cache.pop(cam_id, None)
                fail_count_read = 0
                fail_count_open = 0
                time.sleep(5)

    thread = Thread(target=capture, daemon=True)
    thread.start()

@bp.route('/capture_presence_video/<int:cam_id>')
def capture_presence_video(cam_id):
    camera = Camera_Settings.query.get(cam_id)
    if not camera or not camera.feed_src:
        return "Invalid camera source", 404

    # Mulai thread jika belum jalan
    start_camera_thread(cam_id, camera.feed_src)

    def generate_frames():
        while True:
            frame = None
            if cam_id in camera_frames:
                with camera_locks[cam_id]:
                    frame = camera_frames[cam_id]

            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_presence_logic(camera_id=0):
    # ... (kode Anda, tidak diubah) ...
    # Pastikan semua variabel seperti current_app, face_cascade, dll. tersedia
    camera = Camera_Settings.query.get(camera_id) # Menggunakan query_get dari mock/model Anda
    if not camera:
        return {'success': False, 'message': f'Kamera ID {camera_id} tidak ditemukan.'}

    model_path = get_model_path()
    # personnel_path = get_personnel_folder_path() # personnel_path digunakan di bawah

    if not os.path.exists(model_path):
        return {'success': False, 'message': 'Model tidak tersedia, harap latih model terlebih dahulu.'}
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # Logika load_label_to_name ada di sini di kode Anda, pastikan ini yang diinginkan
    # atau panggil fungsi global load_label_to_name() jika ada
    label_to_name_map = {}
    personnel_path = get_personnel_folder_path() # Pastikan path ini benar
    if os.path.exists(personnel_path) and os.path.isdir(personnel_path):
        for face_folder in os.listdir(personnel_path):
            face_folder_path = os.path.join(personnel_path, face_folder)
            if os.path.isdir(face_folder_path):
                for file_name in os.listdir(face_folder_path):
                    if file_name.lower().endswith(('.jpg','.png','.jpeg')):
                        try:
                            # Asumsi format nama file: prefix_LABEL_NAMALENGKAP_suffix.jpg
                            parts = os.path.splitext(file_name)[0].split('_')
                            if len(parts) >= 3:
                                label = int(parts[1])
                                extracted_name = "_".join(parts[2:]) # Gabungkan sisa nama jika ada underscore
                                label_to_name_map[label] = extracted_name
                        except (IndexError, ValueError) as e:
                            # print(f"Skipping file due to parsing error ({e}): {file_name}")
                            continue
    else:
        print(f"Warning: Personnel path '{personnel_path}' tidak ditemukan atau bukan direktori.")


    cap = None # Inisialisasi
    try:
        camera_feed_source = int(camera.feed_src) if str(camera.feed_src).isdigit() else camera.feed_src
        cap = get_camera_instance(camera_feed_source)
        if not cap or not cap.isOpened():
            return {'success': False, 'message': f'Kamera {camera.name if hasattr(camera, "name") else camera.feed_src} tidak dapat dibuka.'}

        frame_count = 0
        max_frames = 100 # Mungkin perlu disesuaikan
        person_processed_in_this_call = False

        while frame_count < max_frames and not person_processed_in_this_call:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_count += 1
                cv2.waitKey(20) # Beri jeda singkat jika frame gagal dibaca
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if not face_cascade: # Periksa jika face_cascade gagal dimuat
            #     return {'success': False, 'message': 'Face detector tidak siap.'}
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if faces is None or len(faces) == 0:

                frame_count += 1
                cv2.waitKey(20)
                continue

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA) # INTER_AREA mungkin lebih baik untuk downscaling
                label, confidence = recognizer.predict(roi_resized)
                name = label_to_name_map.get(label, "Unknown")
                clean_name = re.sub(r'[_\d]+$', '', name)
                
                # Tambahkan logika kepercayaan di sini jika diperlukan sebelum menyimpan
                # Misalnya: if confidence < MAX_CONFIDENCE_THRESHOLD:
                
                now = datetime.now()
                # Pastikan current_app.config['UPLOAD_FOLDER'] terdefinisi
                upload_folder = current_app.config.get('UPLOAD_FOLDER', os.path.join(current_app.root_path, 'static', 'uploads'))
                save_directory = os.path.join(upload_folder, 'extracted_faces', 'predicted_faces', 'absence', now.strftime('%Y%m%d'))
                os.makedirs(save_directory, exist_ok=True)

                image_name_safe = "".join(c if c.isalnum() or c in ['_'] else '_' for c in clean_name) # Buat nama file aman
                image_path = os.path.join(save_directory, f"{image_name_safe}_{now.strftime('%H%M%S%f')}.jpg")
                
                # Simpan ROI (Region of Interest) wajah yang terdeteksi
                face_image_to_save = frame[y:y+h, x:x+w]
                save_success = cv2.imwrite(image_path, face_image_to_save)

                if not save_success:
                    print(f"Gagal menyimpan gambar ke: {image_path}")
                    # Jangan lanjutkan ke pemrosesan absensi jika gambar gagal disimpan
                    continue # Coba wajah berikutnya jika ada

                data = {
                    'name': clean_name,
                    'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'image_path': image_path, # Atau path relatif jika dibutuhkan
                    'camera_id': camera.id,
                    'confidence': confidence # Sertakan confidence jika berguna untuk proses_attendance_entry
                }

                result = process_attendance_entry(data) # Asumsikan fungsi ini ada
                person_processed_in_this_call = True # Tandai bahwa seseorang telah diproses
                
                # Tidak perlu cap.release() di sini karena akan dirilis di finally atau setelah loop max_frames
                
                # Kembalikan hasil berdasarkan output process_attendance_entry
                if result == 'success':
                    return {'success': True, 'message': f'Absensi tercatat untuk {name}.'}
                elif result == 'already_present': # Sesuaikan string ini dengan output aktual Anda
                    return {'success': False, 'message': f'{name} sudah absen hari ini.'} # "False" agar tidak ada notif sukses berulang
                elif result == 'not_eligible_for_leave':
                    return {'success': False, 'message': 'Tidak dapat mencatat IZIN sebelum TEPAT WAKTU atau TERLAMBAT'}
                elif result == 'personnel_not_found':
                    return {'success': False, 'message': 'Personel tidak ditemukan'}
                elif result == 'invalid_camera':
                     return {'success': False, 'message': 'Kamera tidak valid atau tidak aktif'}
                else: # Handle kasus lain dari process_attendance_entry
                    return {'success': False, 'message': result if isinstance(result, str) else f'Gagal mencatat absensi untuk {name}.'}
            
            if person_processed_in_this_call: # Jika seseorang sudah diproses, keluar dari while loop
                break
            frame_count += 1
        
        # Jika loop selesai tanpa memproses siapa pun
        if not person_processed_in_this_call:
            return {'success': False, 'message': 'Tidak ada wajah terdeteksi/dikenali setelah beberapa percobaan.'}
        # Fallback jika sesuatu yang aneh terjadi
        return {'success': False, 'message': 'Proses absensi selesai dengan status tidak diketahui.'}

    except Exception as e:
        import traceback
        print(f"Error di capture_presence_logic: {e}")
        traceback.print_exc()
        return {'success': False, 'message': f'Terjadi error server: {str(e)}'}
    finally:
        if cap and cap.isOpened():
            # print(f"Melepaskan kamera {camera.feed_src} dari capture_presence_logic.")
            cap.release()


@bp.route('/capture_presence_cam', methods=['POST'])
# @login_required
def capture_presence_cam():
    # *** PERUBAHAN UTAMA DI SINI ***
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request harus berupa JSON.'}), 400

    data = request.get_json()
    if data is None:
        return jsonify({'status': 'error', 'message': 'Tidak ada data JSON diterima.'}), 400

    camera_id_from_request = data.get('camera_id')

    if camera_id_from_request is None: # Jika 'camera_id' tidak ada di JSON
        # Cobalah mencari kamera default jika tidak ada ID yang diberikan
        # Ini sesuai dengan logika lama Anda, tetapi pastikan ini yang diinginkan
        # untuk request JSON
        cam_default = Camera_Settings.query.filter_by(role_camera='P', cam_is_active=True).first()
        if cam_default:
            camera_id_from_request = cam_default.id
            # print(f"Menggunakan kamera default ID: {camera_id_from_request} karena tidak ada di request JSON.")
        else:
            return jsonify({'status': 'error', 'message': 'camera_id tidak ditemukan di request dan tidak ada kamera absensi aktif default.'}), 400
    else:
        try:
            camera_id_from_request = int(camera_id_from_request)
        except ValueError:
            return jsonify({'status': 'error', 'message': 'camera_id harus berupa angka.'}), 400


    result = capture_presence_logic(camera_id_from_request)

    if result.get('success', False): # Default ke False jika 'success' tidak ada
        return jsonify({'status': 'success', 'message': result.get('message', 'Sukses')})
    else:
        # Untuk error "sudah absen" atau "tidak ada wajah", mungkin status HTTP 200 OK
        # tapi dengan status 'info' atau 'error' di JSON lebih baik daripada 500.
        # Kode 500 sebaiknya untuk error server yang tak terduga.
        status_code = 200 # Atau 400/404 tergantung jenis error
        if "tidak ditemukan" in result.get('message','').lower() or \
           "tidak dapat dibuka" in result.get('message','').lower():
            status_code = 404 # Not Found
        
        # Untuk kasus seperti "already_present", "no face detected"
        # kita bisa tetap return 200 OK tapi dengan status error di JSON
        # agar frontend bisa menanganinya sebagai info/peringatan.
        return jsonify({'status': 'error', 'message': result.get('message', 'Terjadi kesalahan')}), status_code if status_code != 200 else 200