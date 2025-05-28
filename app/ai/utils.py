# coorporate_app/app/ai/utils.py
import cv2
import os
from datetime import datetime, date, time, timedelta
from app import db # Import db instance
from app.models import Camera_Settings, Personnels, Personnel_Entries, Work_Timer # Import models
from flask import current_app # Untuk mengakses app.config
from sqlalchemy import func, cast, Date # Untuk fungsi database SQLAlchemy

# Cache untuk instance kamera agar tidak terus-menerus membuka/menutup
_camera_instance_cache = {}

def get_camera_instance(camera_source):
    """
    Mendapatkan instance cv2.VideoCapture.
    Mendukung indeks kamera (0, 1) atau URL (RTSP, HTTP).
    Menggunakan cache untuk menghindari pembukaan berulang.
    """
    if camera_source not in _camera_instance_cache or not _camera_instance_cache[camera_source].isOpened():
        cap = None
        if isinstance(camera_source, int) or camera_source.isdigit():
            # Coba berbagai backend untuk Windows/Linux
            cam_idx = int(camera_source)
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) # DirectShow (Windows)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF) # Media Foundation (Windows)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2) # Video4Linux (Linux)
            if not cap.isOpened(): # Fallback to default if specific backends fail
                cap = cv2.VideoCapture(cam_idx)
        else: # Asumsi URL
            cap = cv2.VideoCapture(camera_source)

        if cap and cap.isOpened():
            _camera_instance_cache[camera_source] = cap
            print(f"Camera instance for source '{camera_source}' opened and cached.")
        else:
            print(f"Error: Could not open camera source '{camera_source}'.")
            return None
    return _camera_instance_cache[camera_source]

def release_camera_instance(camera_source):
    """Melepaskan instance kamera dari cache dan menutupnya."""
    if camera_source in _camera_instance_cache:
        cap = _camera_instance_cache[camera_source]
        if cap.isOpened():
            cap.release()
        del _camera_instance_cache[camera_source]
        print(f"Camera instance for source '{camera_source}' released.")

def generate_simple_camera_frames(camera_source):
    """Generator untuk streaming video mentah tanpa pemrosesan AI."""
    cap = get_camera_instance(camera_source)
    if not cap or not cap.isOpened():
        print(f"Error: Camera at {camera_source} could not be opened for simple stream.")
        return # Keluar dari generator

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_source}. Attempting to re-open.")
            cap.release() # Release the current (failed) capture object
            cap = get_camera_instance(camera_source) # Try to get a new instance
            if not cap or not cap.isOpened():
                print(f"Failed to re-open camera {camera_source}. Stopping stream.")
                break # Exit generator if cannot re-open
            continue # Continue to next iteration to read from new capture

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Pastikan kamera dilepas jika loop berakhir
    release_camera_instance(camera_source)


def process_attendance_entry(data):
    """
    Memproses logika entri kehadiran, diadaptasi untuk Flask-SQLAlchemy.
    """
    name = data.get('name')
    datetime_str = data.get('datetime')
    image_path = data.get('image_path')
    cam_id = data.get('camera_id')
    detected_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    personnel = Personnels.query.filter_by(name=name).first()
    if not personnel:
        print(f"Personnel '{name}' not found for attendance processing.")
        return 'personnel_not_found'
    
    camera_settings = Camera_Settings.query.filter_by(id=cam_id, role_camera=Camera_Settings.ROLE_PRESENCE).first()
    if not camera_settings or not camera_settings.cam_is_active:
        print(f"Camera settings not found or not an active Presence camera for ID {cam_id}.")
        return 'invalid_camera'

    # Fetch existing entries for today for this personnel
    today_start = datetime.combine(detected_time.date(), time.min)
    today_end = datetime.combine(detected_time.date(), time.max)

    existing_entries_today = Personnel_Entries.query.filter(
        Personnel_Entries.personnel_id == personnel.id,
        Personnel_Entries.timestamp.between(today_start, today_end)
    ).all()

    has_ontime = any(e.presence_status == 'ONTIME' for e in existing_entries_today)
    has_late = any(e.presence_status == 'LATE' for e in existing_entries_today)
    has_leave = any(e.presence_status == 'LEAVE' for e in existing_entries_today)

    attendance_start_str = camera_settings.attendance_time_start
    attendance_end_str = camera_settings.attendance_time_end
    leaving_start_str = camera_settings.leaving_time_start
    leaving_end_str = camera_settings.leaving_time_end

    attendance_start = datetime.strptime(attendance_start_str, '%H:%M:%S').time() if attendance_start_str else None
    attendance_end = datetime.strptime(attendance_end_str, '%H:%M:%S').time() if attendance_end_str else None
    leaving_start = datetime.strptime(leaving_start_str, '%H:%M:%S').time() if leaving_start_str else None
    leaving_end = datetime.strptime(leaving_end_str, '%H:%M:%S').time() if leaving_end_str else None

    current_time_only = detected_time.time()

    # Determine status
    status = None
    if attendance_start and attendance_end and attendance_start <= current_time_only <= attendance_end:
        status = 'ONTIME'
    elif leaving_start and leaving_end and leaving_start <= current_time_only <= leaving_end:
        status = 'LEAVE'
    else:
        # Jika tidak masuk kategori ONTIME atau LEAVE, periksa LATE
        if attendance_end and leaving_start and attendance_end < current_time_only < leaving_start:
            status = 'LATE'
        else:
            # Jika di luar semua rentang waktu yang didefinisikan, bisa jadi UNKNOWN atau LEAVE
            # Tergantung pada logika bisnis Anda. Default ke UNKNOWN jika tidak ada aturan lain.
            status = 'UNKNOWN' 

    # Prevent duplicate entries and apply business logic
    if status == 'ONTIME' and has_ontime:
        print("ONTIME already recorded for today.")
        return 'already_present'
    if status == 'LATE' and (has_ontime or has_late):
        print("Either ONTIME or LATE already recorded for today.")
        return 'already_present'
    if status == 'LEAVE':
        if has_leave:
            print("LEAVE already recorded for today.")
            return 'already_present'
        if not (has_ontime or has_late):
            print("Cannot record LEAVE without ONTIME or LATE entry first.")
            return 'not_eligible_for_leave'

    # Jika statusnya LATE tapi sudah ada ONTIME, ubah jadi LEAVE (logika dari Django Anda)
    if status == 'LATE' and has_ontime:
        status = 'LEAVE'
        if has_leave: # Double check if leave already exists after this transition
            return 'already_present'

    # Save into DB
    try:
        new_entry = Personnel_Entries(
            camera_id=cam_id,
            personnel_id=personnel.id,
            timestamp=detected_time, # Gunakan datetime lengkap
            presence_status=status,
            image=image_path
        )
        db.session.add(new_entry)
        db.session.commit()
        print(f"Inserted {status} entry for {name} at {detected_time}")
        return 'success'
    except Exception as e:
        db.session.rollback() # Rollback in case of error
        print(f"Database insertion failed for {name}: {e}")
        return 'db_error'


def capture_absence_from_webcam_logic(personnel_id, camera_id=0):
    """
    Logika untuk mengambil gambar ketidakhadiran dari webcam.
    Ini adalah fungsi inti, bukan view Flask.
    """
    personnel = Personnels.query.get(personnel_id)
    camera = Camera_Settings.query.get(camera_id)

    if not personnel:
        return {"success": False, "message": "Personnel not found."}
    if not camera:
        return {"success": False, "message": "Camera not found."}

    cap = get_camera_instance(int(camera.feed_src) if str(camera.feed_src).isdigit() else camera.feed_src)
    if not cap or not cap.isOpened():
        return {"success": False, "message": "Could not open camera."}

    ret, frame = cap.read()
    # Jangan rilis cap di sini jika get_camera_instance mengelola cache
    # cap.release() # Ini akan menutup kamera untuk semua stream jika dicache

    if not ret:
        return {"success": False, "message": "Failed to capture image from camera."}

    # Save the captured image
    now = datetime.now()
    filename = f"absence_{personnel.name}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    save_directory = os.path.join(current_app.config['UPLOAD_FOLDER'], 'extracted_faces', 'predicted_faces', 'absence', now.strftime('%Y%m%d'))
    os.makedirs(save_directory, exist_ok=True)
    file_path_abs = os.path.join(save_directory, filename)
    
    cv2.imwrite(file_path_abs, frame)

    relative_path_to_upload_folder = os.path.relpath(file_path_abs, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")

    # Record absence entry (status 'UNKNOWN' or 'LEAVE' based on business logic)
    # Panggil process_attendance_entry untuk konsistensi
    data_for_attendance = {
        'name': personnel.name,
        'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
        'image_path': relative_path_to_upload_folder,
        'camera_id': camera.id
    }
    result_status = process_attendance_entry(data_for_attendance)

    if result_status == 'success':
        return {"success": True, "message": f"Absence recorded for {personnel.name}."}
    else:
        return {"success": False, "message": f"Failed to record absence: {result_status}"}