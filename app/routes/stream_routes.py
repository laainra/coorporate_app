# coorporate_app/app/routes/stream_routes.py
import base64
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

    processed_source = int(camera_source) if isinstance(camera_source, str) and camera_source.isdigit() else camera_source

    if processed_source in _camera_instance_cache:
        cap = _camera_instance_cache[processed_source]
        if cap.isOpened():
            return cap
        else:
            print(f"[get_camera_instance] Cached camera for {processed_source} is not opened. Releasing.")
            cap.release()
            del _camera_instance_cache[processed_source]

    cap = None
    tried_backends = []

    if isinstance(processed_source, int):
        # 👇 Prioritaskan backend tercepat
         for backend in [cv2.CAP_MSMF,cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:   # Hindari CAP_MSMF jika tidak perlu
            tried_backends.append(backend)
            cap = cv2.VideoCapture(processed_source, backend)
            if cap.isOpened():
                print(f"[get_camera_instance] Camera opened with backend {backend}.")
                break
    else:
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            tried_backends.append(backend)
            cap = cv2.VideoCapture(processed_source, backend)
            if cap.isOpened():
                print(f"[get_camera_instance] Stream opened with backend {backend}.")
                break

    if cap and cap.isOpened():
        _camera_instance_cache[processed_source] = cap
        return cap
    else:
        print(f"[get_camera_instance] ❌ ERROR: Failed to open camera source '{processed_source}' using backends: {tried_backends}")
        return None


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

# @bp.route('/capture-data', methods=['POST']) # Endpoint untuk menerima data gambar
# @login_required
# def capture_faces(): # Nama fungsi API dibedakan dari nama fungsi view halaman
#     personnel = Personnels.query.filter_by(user_account=current_user).first()
#     if not personnel:
#         return jsonify({'status': 'error', 'message': 'User personnel data not found.'}), 400

#     data = request.get_json()
#     if not data or 'image_data' not in data:
#         return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

#     image_data_url = data['image_data']

#     try:
#         # Ekstrak bagian base64
#         if ',' not in image_data_url:
#             return jsonify({'status': 'error', 'message': 'Invalid image data format (missing comma).'}), 400
#         header, encoded_data = image_data_url.split(',', 1)
#         img_bytes = base64.b64decode(encoded_data)
#     except Exception as e:
#         current_app.logger.error(f"Base64 decoding error: {e}")
#         return jsonify({'status': 'error', 'message': f'Invalid base64 image data: {e}.'}), 400

#     # Konversi bytes ke np array dan decode ke gambar cv2
#     np_arr = np.frombuffer(img_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     if frame is None:
#         return jsonify({'status': 'error', 'message': 'Could not decode image data.'}), 400
    
#     if face_cascade is None: # Periksa apakah cascade sudah dimuat
#         current_app.logger.error("face_cascade tidak dimuat, deteksi wajah tidak bisa dilakukan.")
#         return jsonify({'status': 'error', 'message': 'Face detection module not initialized on server.'}), 500


#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Parameter detectMultiScale mungkin perlu disesuaikan untuk performa/akurasi
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

#     # Dapatkan base path untuk UPLOAD_FOLDER dari konfigurasi aplikasi
#     # Pastikan UPLOAD_FOLDER diset ke path absolut atau path yang benar
#     # dan idealnya berada di dalam folder 'static' Anda jika ingin mudah diakses via URL.
#     # Contoh: UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
#     base_upload_folder = current_app.config.get('UPLOAD_FOLDER')
#     if not base_upload_folder:
#         current_app.logger.error("UPLOAD_FOLDER tidak dikonfigurasi di aplikasi.")
#         return jsonify({'status': 'error', 'message': 'Server UPLOAD_FOLDER not configured.'}), 500

#     # Buat folder khusus untuk personel ini di dalam subfolder 'personnel_images'
#     personnel_images_root = os.path.join(base_upload_folder, "personnel_images")
#     personnel_folder_on_server = os.path.join(personnel_images_root, str(personnel.id) + "_" + "".join(c if c.isalnum() else "_" for c in personnel.name) ) # Nama folder lebih aman
    
#     try:
#         os.makedirs(personnel_folder_on_server, exist_ok=True)
#     except OSError as e:
#         current_app.logger.error(f"Gagal membuat direktori {personnel_folder_on_server}: {e}")
#         return jsonify({'status': 'error', 'message': 'Could not create directory for images on server.'}), 500


#     existing_images_count_db = Personnel_Images.query.filter_by(personnel_id=personnel.id).count()
#     captured_faces_this_frame = 0

#     for (x, y, w, h) in faces:
#         face_roi = frame[y:y+h, x:x+w] # Simpan ROI berwarna, bukan grayscale, untuk kualitas dataset
#         if face_roi.size > 0:
#             existing_images_count_db += 1 # Increment untuk nama file unik
#             # Buat nama file yang lebih aman dan informatif
#             filename = f"face_{personnel.id}_{existing_images_count_db}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
#             filepath_on_server = os.path.join(personnel_folder_on_server, filename)
            
#             try:
#                 cv2.imwrite(filepath_on_server, face_roi)
#                 current_app.logger.info(f"Wajah disimpan ke: {filepath_on_server}")

#                 # Buat path relatif terhadap UPLOAD_FOLDER untuk disimpan di DB
#                 # Ini penting agar url_for('static', filename=...) bisa bekerja.
#                 # relative_path_to_upload_folder = os.path.relpath(filepath_on_server, base_upload_folder).replace("\\", "/")
                
#                 # Cara yang lebih aman untuk path relatif jika UPLOAD_FOLDER adalah root untuk static/uploads
#                 # dan personnel_images adalah subfolder dari UPLOAD_FOLDER/personnel_images
#                 # Path yang disimpan di DB harus relatif terhadap folder static Flask Anda, bukan UPLOAD_FOLDER
#                 # Asumsikan UPLOAD_FOLDER = 'app/static/uploads'
#                 # maka relative_path akan seperti 'uploads/personnel_images/id_nama/face_....jpg'
#                 # yang bisa digunakan di url_for('static', filename=relative_path)
                
#                 # Cara membuat path relatif yang benar:
#                 # 1. Dapatkan path absolut ke folder static Flask
#                 static_folder_abs = os.path.abspath(current_app.static_folder)
#                 # 2. Dapatkan path absolut file yang disimpan
#                 filepath_abs = os.path.abspath(filepath_on_server)

#                 if filepath_abs.startswith(static_folder_abs):
#                     relative_path_for_db = os.path.relpath(filepath_abs, static_folder_abs).replace("\\", "/")
#                     new_image = Personnel_Images(personnel_id=personnel.id, image_path=relative_path_for_db)
#                     db.session.add(new_image)
#                     captured_faces_this_frame += 1
#                 else:
#                     current_app.logger.error(f"File yang disimpan {filepath_abs} tidak berada di dalam static folder {static_folder_abs}. Tidak bisa membuat path relatif untuk DB.")

#             except Exception as e_write:
#                 current_app.logger.error(f"Gagal menyimpan file gambar {filepath_on_server}: {e_write}")
#                 # Lanjutkan ke wajah berikutnya jika ada
#                 continue


#     if captured_faces_this_frame > 0:
#         try:
#             db.session.commit()
#             return jsonify({'status': 'success', 
#                             'message': f'{captured_faces_this_frame} wajah berhasil diambil dan disimpan.', 
#                             'face_saved_this_frame': True,
#                             'total_images_for_personnel': existing_images_count_db
#                             })
#         except Exception as e_commit:
#             db.session.rollback()
#             current_app.logger.error(f"Gagal commit ke database: {e_commit}")
#             return jsonify({'status': 'error', 'message': 'Gagal menyimpan data gambar ke database.'}), 500
#     else:
#         return jsonify({'status': 'success', 'message': 'Tidak ada wajah terdeteksi di frame ini.', 'face_saved_this_frame': False})

# Variabel global tracking waktu (optional)
last_detection_time = {}
detection_times = {}

@bp.route('/capture_video')
@login_required
def capture_video():


    def generate_frames():
 

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
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
    failed_read_count = 0
    max_failed_reads = 5  # Jumlah maksimal gagal read sebelum reconnect
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Haar Cascade model gagal dimuat.")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "MODEL ERROR!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', dummy_frame)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                print(f"[Stream] Failed to read frame from {camera_source}. Attempting to recover... ({failed_read_count + 1})")
                failed_read_count += 1
                time.sleep(0.05)  # beri waktu CPU istirahat sedikit

                if failed_read_count >= max_failed_reads:
                    print(f"[Stream] Reconnecting stream for {camera_source} after {max_failed_reads} failed reads.")
                    cap.release()
                    time.sleep(1)  # beri delay sebelum reconnect

                    cap = get_camera_instance(int(camera_source) if str(camera_source).isdigit() else camera_source)
                    if not cap or not cap.isOpened():
                        print(f"[Stream] Reconnect failed. Stopping stream for {camera_source}.")
                        break
                    failed_read_count = 0  # reset hitungan gagal
                continue

            failed_read_count = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray is None or gray.size == 0:
                print("Warning: Frame kosong, skip detection")
                continue  # Skip iterasi ini

            try:
                faces_detected_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            except cv2.error as e:
                print(f"OpenCV detectMultiScale error: {e}")
                continue  # Skip iterasi ini

            if len(faces_detected_in_frame) == 0:
                pass

            for (x, y, w, h) in faces_detected_in_frame:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label, confidence = recognizer.predict(face_roi)

                name = label_to_name.get(label, "Unknown")
                timer_text = ""
                display_color = (0, 0, 255)

                if confidence < 60:
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
        except Exception as e:
            print(f"[Stream Error] {e}")
            break  # atau continue jika mau tetap lanjut
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
    
@bp.route('/quick_preview/<int:cam_id>')
def quick_preview(cam_id):
    def gen():
        print(f"[Quick Preview] Starting camera preview for {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            yield b"Failed to open camera."
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1 / 15)

        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/capture_presence_video/<int:cam_id>')
def capture_presence_video(cam_id):
    camera = Camera_Settings.query.get(cam_id)
    if not camera or not camera.feed_src:
        # Mengirim respons error yang bisa ditangani klien jika perlu,
        # atau biarkan browser menampilkan gambar rusak yang akan memicu onerror.
        # Untuk multipart, lebih baik menghentikan stream jika sumber tidak valid.
        # Kita bisa mengirim satu frame error dan kemudian berhenti.
        error_msg = "Invalid camera source"
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8) # Frame kecil untuk error
        cv2.putText(dummy_frame, error_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', dummy_frame)
        if ret:
            return Response(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n',
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        return "Invalid camera source", 404


    # Mulai thread jika belum jalan atau pastikan thread untuk cam_id ini aktif
    start_camera_thread(cam_id, camera.feed_src)

    def generate_frames_with_timeout():
        last_valid_frame_time = time.time()
        NO_FRAME_TIMEOUT = 10  # Detik. Jika tidak ada frame baru selama ini, anggap stream mati.

        while True:
            current_time = time.time()
            frame_to_send = None

            if cam_id in camera_frames:
                with camera_locks[cam_id]:
                    # Ambil frame hanya jika ada dan merupakan numpy array
                    if isinstance(camera_frames[cam_id], np.ndarray):
                        frame_to_send = camera_frames[cam_id].copy() # Kirim copy untuk thread safety

            if frame_to_send is not None:
                last_valid_frame_time = current_time
                ret, jpeg = cv2.imencode('.jpg', frame_to_send)
                if ret:
                    try:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    except Exception as e: # Misal client disconnect
                        print(f"[Generator {cam_id}] Error yielding frame: {e}")
                        break 
                else:
                    print(f"[Generator {cam_id}] Failed to encode frame.")
                    # Pertimbangkan untuk mengirim frame error di sini juga atau break
            else:
                # Tidak ada frame valid yang tersedia dari thread kamera
                if (current_time - last_valid_frame_time) > NO_FRAME_TIMEOUT:
                    print(f"[Generator {cam_id}] No valid frame received for {NO_FRAME_TIMEOUT}s. Stopping stream yield.")
                    # Mengirim frame "NO SIGNAL" atau error sebelum break
                    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(dummy_frame, "NO SIGNAL", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, jpeg = cv2.imencode('.jpg', dummy_frame)
                    if ret:
                        try:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                        except Exception as e:
                            print(f"[Generator {cam_id}] Error yielding NO SIGNAL frame: {e}")
                    break # Hentikan generator, ini akan memutus koneksi klien & memicu img.onerror

            time.sleep(1 / 20) # Target sekitar 20 FPS untuk generator ini, sesuaikan dengan FPS_TARGET di thread capture

        print(f"[Generator {cam_id}] Frame generation stopped.")

    return Response(generate_frames_with_timeout(), mimetype='multipart/x-mixed-replace; boundary=frame')
MAX_CONFIDENCE_THRESHOLD = 70  # Contoh nilai, sesuaikan!

def capture_presence_logic(camera_id=0):
    # ... (kode awal Anda untuk mengambil camera, model_path, recognizer, label_to_name_map tetap sama) ...
    # Pastikan semua variabel seperti current_app, face_cascade, dll. tersedia
    camera = Camera_Settings.query.get(camera_id)
    if not camera:
        return {'success': False, 'message': f'Kamera ID {camera_id} tidak ditemukan.'}

    model_path = get_model_path()
    if not os.path.exists(model_path):
        return {'success': False, 'message': 'Model tidak tersedia, harap latih model terlebih dahulu.'}
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    label_to_name_map = {}
    personnel_path = get_personnel_folder_path() 
    if os.path.exists(personnel_path) and os.path.isdir(personnel_path):
        for face_folder in os.listdir(personnel_path):
            face_folder_path = os.path.join(personnel_path, face_folder)
            if os.path.isdir(face_folder_path):
                for file_name in os.listdir(face_folder_path):
                    if file_name.lower().endswith(('.jpg','.png','.jpeg')):
                        try:
                            parts = os.path.splitext(file_name)[0].split('_')
                            if len(parts) >= 3:
                                label = int(parts[1])
                                extracted_name = "_".join(parts[2:])
                                label_to_name_map[label] = extracted_name
                        except (IndexError, ValueError) as e:
                            continue
    else:
        print(f"Warning: Personnel path '{personnel_path}' tidak ditemukan atau bukan direktori.")

    cap = None 
    try:
        camera_feed_source = int(camera.feed_src) if str(camera.feed_src).isdigit() else camera.feed_src
        cap = get_camera_instance(camera_feed_source)
        if not cap or not cap.isOpened():
            return {'success': False, 'message': f'Kamera {camera.name if hasattr(camera, "name") else camera.feed_src} tidak dapat dibuka.'}

        frame_count = 0
        max_frames = 100 
        person_processed_in_this_call = False
        face_detected_with_sufficient_confidence = False # Flag baru

        while frame_count < max_frames and not person_processed_in_this_call:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_count += 1
                cv2.waitKey(20) 
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Pastikan face_cascade sudah diinisialisasi dengan benar sebelumnya
            # Misalnya: face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if 'face_cascade' not in globals() or face_cascade.empty():
                 print("Error: Haar Cascade model gagal dimuat atau tidak ada.")
                 return {'success': False, 'message': 'Model detektor wajah tidak siap.'}

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if faces is None or len(faces) == 0:
                frame_count += 1
                cv2.waitKey(20)
                continue

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA)
                label, confidence = recognizer.predict(roi_resized)
                
                # Cetak confidence untuk membantu menentukan threshold yang tepat saat testing
                print(f"Detected Label: {label}, Name: {label_to_name_map.get(label, 'Unknown')}, Confidence: {confidence:.2f}")

                # --- PENGECEKAN CONFIDENCE DIMULAI DI SINI ---
                if confidence < MAX_CONFIDENCE_THRESHOLD:
                    name = label_to_name_map.get(label, "Unknown")
                    if name == "Unknown":
                        # Jika label tidak ada di map, atau confidence terlalu tinggi meskipun dikenali sebagai label tertentu
                        # Anda bisa memilih untuk tidak memprosesnya sebagai "dikenali"
                        print(f"Wajah terdeteksi sebagai 'Unknown' atau label tidak ada di map, confidence: {confidence:.2f}. Tidak diproses.")
                        # Jika ingin tetap mencoba wajah lain dalam frame yang sama, gunakan 'continue'
                        # Jika ingin menghentikan pemrosesan frame ini dan lanjut ke frame berikutnya,
                        # ini sudah ditangani oleh loop 'for (x,y,w,h) in faces:'
                        continue # Lanjut ke wajah berikutnya jika ada, atau frame berikutnya jika tidak

                    clean_name = re.sub(r'[_\d]+$', '', name)
                    face_detected_with_sufficient_confidence = True # Tandai bahwa wajah yang cukup dikenal terdeteksi
                    print(f"Wajah dikenali sebagai: {clean_name} dengan confidence: {confidence:.2f} (DI BAWAH THRESHOLD {MAX_CONFIDENCE_THRESHOLD})")

                    # --- PROSES PENYIMPANAN DAN ABSENSI HANYA JIKA CONFIDENCE CUKUP ---
                    now = datetime.now()
                    upload_folder = current_app.config.get('UPLOAD_FOLDER', os.path.join(current_app.root_path, 'static', 'uploads'))
                    save_directory = os.path.join(upload_folder, 'extracted_faces', 'predicted_faces', 'absence', now.strftime('%Y%m%d'))
                    os.makedirs(save_directory, exist_ok=True)

                    image_name_safe = "".join(c if c.isalnum() or c in ['_'] else '_' for c in clean_name)
                    image_path = os.path.join(save_directory, f"{image_name_safe}_{now.strftime('%H%M%S%f')}_{int(confidence)}.jpg") # Tambahkan confidence ke nama file
                    
                    face_image_to_save = frame[y:y+h, x:x+w]
                    save_success = cv2.imwrite(image_path, face_image_to_save)

                    if not save_success:
                        print(f"Gagal menyimpan gambar ke: {image_path}")
                        continue 

                    data = {
                        'name': clean_name,
                        'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
                        'image_path': image_path, 
                        'camera_id': camera.id,
                        'confidence': float(confidence) # Simpan confidence sebagai float
                    }

                    result = process_attendance_entry(data) 
                    person_processed_in_this_call = True 
                    
                    parts = name.rsplit('_', 1)
                    display_name = name 
                    if len(parts) == 2 and parts[1].isdigit():
                        display_name = parts[0]
                    
                    if result == 'success':
                        return {'success': True, 'message': f'Absensi tercatat untuk {display_name} (Conf: {confidence:.2f}).'}
                    # ... (sisa penanganan 'result' Anda) ...
                    elif result == 'already_present':
                        return {'success': False, 'message': f'{display_name} sudah absen hari ini.'}
                    elif result == 'not_eligible_for_leave':
                        return {'success': False, 'message': 'Tidak dapat mencatat IZIN sebelum TEPAT WAKTU atau TERLAMBAT'}
                    elif result == 'personnel_not_found':
                        return {'success': False, 'message': 'Personel tidak ditemukan'}
                    elif result == 'invalid_camera':
                        return {'success': False, 'message': 'Kamera tidak valid atau tidak aktif'}
                    else: 
                        return {'success': False, 'message': result if isinstance(result, str) else f'Gagal mencatat absensi untuk {name} (Conf: {confidence:.2f}).'}

                else: # Jika confidence TIDAK memenuhi syarat (terlalu tinggi)
                    print(f"Wajah terdeteksi (Label: {label}, Name: {label_to_name_map.get(label, 'Unknown')}) tetapi confidence {confidence:.2f} terlalu tinggi (DI ATAS THRESHOLD {MAX_CONFIDENCE_THRESHOLD}). Tidak diproses.")
                    # Anda bisa memutuskan untuk tidak melakukan apa-apa, atau mencatatnya sebagai "tidak dikenali"
                    # Untuk saat ini, kita akan lanjut ke deteksi wajah berikutnya dalam frame yang sama jika ada

            if person_processed_in_this_call: 
                break 
            frame_count += 1
        
        # Jika loop selesai
        if not person_processed_in_this_call:
            if face_detected_with_sufficient_confidence: # Pernah ada wajah dikenali tapi mungkin gagal di proses_attendance_entry
                 return {'success': False, 'message': 'Wajah dikenali tetapi gagal diproses untuk absensi.'}
            else: # Tidak ada wajah yang dikenali dengan confidence cukup
                 return {'success': False, 'message': 'Tidak ada wajah terdeteksi/dikenali dengan keyakinan cukup setelah beberapa percobaan.'}
        
        return {'success': False, 'message': 'Proses absensi selesai dengan status tidak diketahui.'}

    except cv2.error as cv_err:
        import traceback
        print(f"OpenCV Error di capture_presence_logic: {cv_err}. Args: {cv_err.args}")
        traceback.print_exc()
        return {'success': False, 'message': f'Terjadi error OpenCV: {str(cv_err)}'}
    except Exception as e:
        import traceback 
        print(f"Error di capture_presence_logic: {e}")
        traceback.print_exc()
        return {'success': False, 'message': f'Terjadi error server: {str(e)}'}
    finally:
        if cap and cap.isOpened():
            cap.release()
            print(f"Kamera {camera.feed_src if hasattr(camera, 'feed_src') else camera_id} dilepas dari capture_presence_logic.")

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
    
@bp.route('/process_frame', methods=['POST'])
@login_required
def process_frame():
    """
    Endpoint untuk menerima frame dari frontend,
    deteksi wajah dilakukan di backend (kalau ada),
    atau minimal simpan gambar frame ke folder personnel,
    dan simpan pathnya ke database.
    """

    data = request.get_json()
    image_data = data.get('image_data', None)
    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

    # Parse image base64 data URL "data:image/jpeg;base64,/9j/...."
    try:
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Invalid image data: {str(e)}'}), 400

    # Get personnel user
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        return jsonify({'status': 'error', 'message': 'Personnel data not found.'}), 400

    # Save image file (e.g. jpg) ke folder personnel
    personnel_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'personnel', str(personnel.id))
    os.makedirs(personnel_folder, exist_ok=True)

    # Buat nama file unik, misal timestamp
    import time
    file_name = f"face_{personnel.id}_{int(time.time() * 1000)}.jpg"
    file_path = os.path.join(personnel_folder, file_name)

    with open(file_path, 'wb') as f:
        f.write(img_bytes)

    # Simpan ke DB relative path
    relative_path = os.path.relpath(file_path, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
    new_image = Personnel_Images(personnel_id=personnel.id, image_path=relative_path)
    db.session.add(new_image)
    db.session.commit()

    # Untuk demo, kita anggap selalu berhasil simpan 1 face
    return jsonify({'status': 'success', 'message': 'Face image saved.', 'face_saved_this_frame': True})

@bp.route('/ping_server_health_check')
def ping_server():
    """Endpoint sederhana untuk memeriksa apakah server aktif."""
    return jsonify(status="ok", message="Server is alive."), 200