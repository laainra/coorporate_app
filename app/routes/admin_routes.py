from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from flask_login import login_required, current_user, logout_user
from app.models import Company, Divisions, Personnels, Personnel_Entries, Camera_Settings, Work_Timer # Import Work_Timer
from app import db # Import instance db
from app.utils.decorators import admin_required # Decorators
from datetime import datetime, date, timedelta # Import timedelta
from sqlalchemy import func, cast, Date, and_, text
import io
import pandas as pd
from werkzeug.utils import secure_filename # Untuk upload file
import os
import shutil
from collections import defaultdict # Untuk defaultdict


bp = Blueprint('admin', __name__, template_folder='../templates/admin_panel')

@bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account is not linked to any company. Please contact superadmin.", "danger")
        logout_user() # Atau redirect ke halaman error
        return redirect(url_for('auth.login'))
        
    # current_month = datetime.now().month
    # current_year = datetime.now().year

    # # Summary statistics for the current month
    # # Adaptasi dari Django ORM ke SQLAlchemy
    # summary_ontime = Personnel_Entries.query.join(Personnels).filter(
    #     Personnels.company_obj == company,
    #     func.month(Personnel_Entries.timestamp) == current_month,
    #     func.year(Personnel_Entries.timestamp) == current_year,
    #     Personnel_Entries.presence_status == 'ONTIME'
    # ).count()

    # summary_late = Personnel_Entries.query.join(Personnels).filter(
    #     Personnels.company_obj == company,
    #     func.month(Personnel_Entries.timestamp) == current_month,
    #     func.year(Personnel_Entries.timestamp) == current_year,
    #     Personnel_Entries.presence_status == 'LATE'
    # ).count()

    # summary_unknown = Personnel_Entries.query.join(Personnels).filter(
    #     Personnels.company_obj == company,
    #     func.month(Personnel_Entries.timestamp) == current_month,
    #     func.year(Personnel_Entries.timestamp) == current_year,
    #     Personnel_Entries.presence_status == 'UNKNOWN'
    # ).count()
    
    # summary = {
    #     'ontime': summary_ontime,
    #     'late': summary_late,
    #     'unknown': summary_unknown,
    # }

    # # Fetch Personnel_Entries data for today
    # today = date.today()
    # presence_data_query = Personnel_Entries.query.join(Personnels).filter(
    #     Personnels.company_obj == company,
    #     cast(Personnel_Entries.timestamp, Date) == today # Filter by date only
    # ).order_by(Personnel_Entries.timestamp.desc()).all()

    # presence_data = []
    # for entry in presence_data_query:
    #     # Perlu memastikan personnel_obj dan camera_obj ada
    #     personnel_name = entry.personnel_obj.name if entry.personnel_obj else 'N/A'
    #     camera_name = entry.camera_obj.cam_name if entry.camera_obj else 'N/A'
    #     presence_data.append({
    #         'id': entry.id,
    #         'name': personnel_name,
    #         'presence_status': entry.presence_status,
    #         'timestamp': entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
    #         'camera_name': camera_name
    #     })


    # # Fetch top employees based on on-time Personnel_Entries
    # # Subquery for Count in SQLAlchemy
    # from sqlalchemy import func
    # top_employees_query = db.session.query(
    #     Personnels.id,
    #     Personnels.name,
    #     Divisions.name.label('division_name'),
    #     func.count(Personnel_Entries.id).label('total_ontime')
    # ).join(Divisions, Personnels.division_id == Divisions.id)\
    #  .join(Personnel_Entries, Personnels.id == Personnel_Entries.personnel_id)\
    #  .filter(
    #     Personnels.company_obj == company,
    #     func.month(Personnel_Entries.timestamp) == current_month,
    #     func.year(Personnel_Entries.timestamp) == current_year,
    #     Personnel_Entries.presence_status == 'ONTIME'
    # ).group_by(Personnels.id, Personnels.name, Divisions.name)\
    #  .order_by(func.count(Personnel_Entries.id).desc())\
    #  .limit(5).all()

    # top_employees = []
    # for emp in top_employees_query:
    #     top_employees.append({
    #         'id': emp.id,
    #         'name': emp.name,
    #         'division': emp.division_name,
    #         'total_ontime': emp.total_ontime
    #     })
        
    summary = {
        'ontime': 120,  # Dummy count of on-time entries
        'late': 30,     # Dummy count of late entries
        'unknown': 5,   # Dummy count of unknown entries
    }

    # Dummy presence data for today
    today = datetime.now().date()
    presence_data = [
        {'id': 'E001', 'name': 'Alice', 'presence_status': 'ontime', 'timestamp': today},
        {'id': 'E002', 'name': 'Bob', 'presence_status': 'late', 'timestamp': today},
        {'id': 'E003', 'name': 'Charlie', 'presence_status': 'ontime', 'timestamp': today},
        {'id': 'E004', 'name': 'David', 'presence_status': 'unknown', 'timestamp': today},
        {'id': 'E005', 'name': 'Eve', 'presence_status': 'ontime', 'timestamp': today},
    ]

    # Dummy top employees based on on-time Personnel_Entries
    top_employees = [
        {'id': 'E001', 'name': 'Alice', 'division': 'HR', 'total_ontime': 20},
        {'id': 'E002', 'name': 'Bob', 'division': 'IT', 'total_ontime': 15},
        {'id': 'E003', 'name': 'Charlie', 'division': 'Finance', 'total_ontime': 25},
        {'id': 'E004', 'name': 'David', 'division': 'Marketing', 'total_ontime': 10},
        {'id': 'E005', 'name': 'Eve', 'division': 'Sales', 'total_ontime': 5},
    ]

    return render_template('admin_panel/dashboard.html',
                           company=company,
                           summary=summary,
                           presence_data=presence_data,
                           top_employees=top_employees)

@bp.route('/divisions')
@login_required
@admin_required
def division():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))
    
    divisions = Divisions.query.filter_by(company_obj=company).all()
    division_count = len(divisions)
    return render_template('admin_panel/division.html', divisions=divisions, division_count=division_count)

@bp.route('/get_divisions')
@login_required
@admin_required
def get_divisions():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        return jsonify({'status': 'error', 'message': 'User does not belong to any company'}), 403
    
    divisions = Divisions.query.filter_by(company_obj=company).all()
    divisions_data = [{'id': div.id, 'name': div.name} for div in divisions]
    return jsonify({'status': 'success', 'divisions': divisions_data})

@bp.route('/add_division', methods=['POST'])
@login_required
@admin_required
def add_division():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        return jsonify({'status': 'error', 'message': 'User does not belong to any company'}), 403

    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return jsonify({'status': 'error', 'message': 'Division name is required'}), 400

        existing_division = Divisions.query.filter_by(name=name, company_obj=company).first()
        if existing_division:
            return jsonify({'status': 'error', 'message': 'Division with this name already exists'}), 409

        try:
            new_division = Divisions(name=name, company_obj=company)
            db.session.add(new_division)
            db.session.commit()
            flash('Division created successfully', 'success') # Flash message ini tidak akan muncul karena reload
            return jsonify({'status': 'success', 'message': 'Division created successfully'}) # <--- PENTING: Pastikan ini dikembalikan
        except Exception as e:
            db.session.rollback()
            # Sangat penting untuk mengembalikan status error yang jelas
            return jsonify({'status': 'error', 'message': f'Failed to add division: {str(e)}'}), 500

    # Pastikan jika request method bukan POST, ada response yang valid
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@bp.route('/edit_division/<int:division_id>', methods=['POST'])
@login_required
@admin_required
def edit_division(division_id):
    division = Divisions.query.get_or_404(division_id)
    # Periksa kepemilikan company
    if division.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to division'}), 403

    if request.method == 'POST':
        new_name = request.form.get('name')
        if not new_name:
            return jsonify({'status': 'error', 'message': 'Division name is required'}), 400

        existing_division = Divisions.query.filter(
            Divisions.name == new_name,
            Divisions.company_obj == division.company_obj,
            Divisions.id != division_id
        ).first()
        if existing_division:
            return jsonify({'status': 'error', 'message': 'Another division with this name already exists'}), 409

        division.name = new_name
        db.session.commit()
        flash('Division updated successfully', 'success')
        return jsonify({'status': 'success', 'message': 'Division updated successfully'})
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@bp.route('/delete_division/<int:division_id>', methods=['POST'])
@login_required
@admin_required
def delete_division(division_id):
    division = Divisions.query.get_or_404(division_id)
    # Periksa kepemilikan company
    if division.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to division'}), 403
    
    try:
        db.session.delete(division)
        db.session.commit()
        flash('Division deleted successfully', 'success')
        return jsonify({'status': 'success', 'message': 'Division deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Failed to delete division: {str(e)}'}), 500

@bp.route('/get_division/<int:division_id>')
@login_required
@admin_required
def get_division(division_id):
    division = Divisions.query.get_or_404(division_id)
    # Periksa kepemilikan company
    if division.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to division'}), 403
        
    division_data = {'id': division.id, 'name': division.name}
    return jsonify({'status': 'success', 'division': division_data})

@bp.route('/employees')
@login_required
@admin_required
def employees():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))
        
    personnels_list = Personnels.query.filter_by(company_obj=company).all()
    divisions_list = Divisions.query.filter_by(company_obj=company).all()
    return render_template('admin_panel/employees.html', employees=personnels_list, divisions=divisions_list)

# @bp.route('/presence')
# @login_required
# @admin_required
# def presence():
#     company = Company.query.filter_by(user_id=current_user.id).first()
#     if not company:
#         flash("Admin account not linked to a company.", "danger")
#         return redirect(url_for('auth.login'))

#     date_str = request.args.get('date')
#     personnel_id = request.args.get('personnel_id', type=int)
    
#     selected_date = date.today()
#     if date_str:
#         try:
#             selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
#         except ValueError:
#             flash("Invalid date format. Using today's date.", "warning")

#     # Filter kehadiran berdasarkan perusahaan admin yang login
#     base_query = Personnel_Entries.query.join(Personnels).filter(
#         Personnels.company_obj == company,
#         cast(Personnel_Entries.timestamp, Date) == selected_date
#     )

#     if personnel_id:
#         base_query = base_query.filter(Personnel_Entries.personnel_id == personnel_id)

#     presence_data = base_query.order_by(Personnel_Entries.timestamp.desc()).all()
#     personnel_list = Personnels.query.filter_by(company_obj=company).all()

#     context = {
#         'presence_data': presence_data,
#         'personnel_list': personnel_list,
#         'selected_date': selected_date.isoformat(), # Format tanggal ke string standar ISO
#         'selected_personnel': personnel_id,
        
#     }
#     print(context)
#     return render_template('admin_panel/presence.html', **context, )

@bp.route('/presence_cam')
@login_required
@admin_required
def presence_cam():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))
    
    # Filter kamera berdasarkan role 'P' dan perusahaan
    cams = Camera_Settings.query.filter(
        Camera_Settings.company_obj == company,
        Camera_Settings.role_camera == Camera_Settings.ROLE_PRESENCE, # Only 'P' as per current models.
        # Q(role_camera='P_IN') | Q(role_camera='P_OUT') from Django implies more specific roles.
        # If your model only has 'P' and 'T', you might need to refine the logic or add P_IN/P_OUT to model choices.
    ).all()
    return render_template(
        'admin_panel/presence_cam.html', 
        cams=cams,
        hasattr=hasattr,      # <--- TAMBAHKAN INI
   
    )


@bp.route('/tracking_cam')
@login_required
@admin_required
def tracking_cam():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))

    cams = Camera_Settings.query.filter(
        Camera_Settings.role_camera == Camera_Settings.ROLE_TRACKING, 
        Camera_Settings.company_obj == company
    ).all()
    return render_template('admin_panel/tracking_cam.html', cams=cams)

@bp.route('/work_time_report') # Pastikan URL ini benar
@login_required
# @admin_required
def work_time_report():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    personnel_list = Personnels.query.filter_by(company_id=company.id).order_by(Personnels.name).all()

    filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
    try:
        filter_date_obj = datetime.strptime(filter_date_str, '%Y-%m-%d').date()
    except ValueError:
        flash("Format tanggal tidak valid. Menggunakan tanggal hari ini.", "warning")
        filter_date_str = date.today().strftime('%Y-%m-%d')
        filter_date_obj = date.today()

    filter_personnel_id_str = request.args.get('filter_personnel_id')

    # --- AWAL PERUBAHAN QUERY ---
    # Query ini akan mengambil entri work_timer dengan nilai timer tertinggi
    # untuk setiap personel pada tanggal yang difilter.
    # Kita menggunakan subquery atau window function untuk ini.
    # Window function (ROW_NUMBER() OVER (...)) lebih modern dan efisien.

    # PERHATIAN: Sintaks window function mungkin sedikit berbeda antar versi MySQL/MariaDB.
    # Ini untuk MySQL 8.0+ / MariaDB 10.2+
    # Jika menggunakan versi lebih lama, pendekatan subquery JOIN mungkin diperlukan.

    sql_query = """
    WITH RankedWorkTimer AS (
        SELECT 
            wt.id AS work_timer_id, 
            wt.datetime AS entry_datetime, 
            wt.type AS timer_type, 
            wt.timer AS timer_seconds, 
            wt.camera_id, 
            wt.personnel_id,
            p.name AS employee_name, 
            d.name AS employee_division, 
            cs.cam_name AS camera_name,
            p.id AS employee_internal_id, 
            -- Jika ada NIP: p.nomor_induk_pegawai AS employee_display_id,
            ROW_NUMBER() OVER(PARTITION BY wt.personnel_id ORDER BY wt.timer DESC, wt.datetime DESC) as rn
        FROM 
            work_timer wt
        JOIN 
            personnels p ON wt.personnel_id = p.id
        JOIN 
            camera_settings cs ON wt.camera_id = cs.id
        JOIN 
            divisions d ON p.division_id = d.id
        WHERE 
            DATE(wt.datetime) = :filter_date AND p.company_id = :company_id 
            {personnel_filter} -- Placeholder untuk filter personel
    )
    SELECT 
        work_timer_id, entry_datetime, timer_type, timer_seconds, camera_id, personnel_id,
        employee_name, employee_division, camera_name, employee_internal_id
        -- Jika ada NIP: , employee_display_id
    FROM RankedWorkTimer
    WHERE rn = 1
    ORDER BY employee_name;
    """
    
    personnel_filter_sql = ""
    params = {'filter_date': filter_date_obj, 'company_id': company.id}

    if filter_personnel_id_str and filter_personnel_id_str.isdigit():
        personnel_filter_sql = "AND p.id = :personnel_id"
        params['personnel_id'] = int(filter_personnel_id_str)
    
    final_sql_query = sql_query.format(personnel_filter=personnel_filter_sql)
    # --- AKHIR PERUBAHAN QUERY ---

    try:
        result_proxy = db.session.execute(text(final_sql_query), params)
        # Setiap baris sudah merupakan data 'timer terakhir' per karyawan
        last_timer_entries = [row._asdict() for row in result_proxy]
    except Exception as e:
        flash(f"Terjadi error saat mengambil data laporan: {e}", "danger")
        print(f"SQL Query Error: {e}")
        print(f"Query: {final_sql_query}")
        print(f"Params: {params}")
        last_timer_entries = []

    report_data_final = []
    # Karena query sudah mengambil timer terakhir per karyawan, agregasi tidak diperlukan lagi
    # Kita hanya perlu memformat data untuk ditampilkan
    for entry_dict in last_timer_entries:
        total_seconds_val = entry_dict.get('timer_seconds', 0)
        total_hours = total_seconds_val // 3600
        remaining_seconds_after_hours = total_seconds_val % 3600
        total_minutes = remaining_seconds_after_hours // 60
        final_seconds = remaining_seconds_after_hours % 60
        
        # Kumpulkan semua area CCTV terkait dengan entri timer terakhir ini
        # Jika satu karyawan punya timer terakhir yang sama dari beberapa kamera (jarang terjadi jika ada datetime DESC),
        # query di atas akan memilih salah satunya. Untuk mengambil semua area CCTV dari semua entri karyawan
        # pada hari itu, kita perlu query tambahan atau modifikasi query yang lebih kompleks.
        # Untuk kesederhanaan, kita ambil camera_name dari entri timer terakhir ini.
        cctv_area_for_this_entry = entry_dict.get('camera_name', 'N/A')

        report_data_final.append({
            # Gunakan ID internal personnel atau NIP jika ada di query
            'employee_id': entry_dict.get('employee_internal_id', 'N/A'), 
            'employee_name': entry_dict.get('employee_name', 'N/A'),
            'division': entry_dict.get('employee_division', 'N/A'),
            'total_time_hours': total_hours,
            'total_time_minutes': total_minutes,
            'total_time_seconds': final_seconds,
            'cctv_areas': cctv_area_for_this_entry, # Hanya dari entri timer terakhir
            'date': entry_dict.get('entry_datetime').date() if entry_dict.get('entry_datetime') else filter_date_obj,
        })
        
    # Urutkan jika belum diurutkan oleh query (query sudah ada ORDER BY employee_name)
    # report_data_final.sort(key=lambda x: (x['employee_name'] if x['employee_name'] else ""))

    return render_template('admin_panel/work_time_report.html', 
                           personnel_list=personnel_list,
                           work_time_report=report_data_final,
                           filter_date=filter_date_str, 
                           filter_personnel_id=int(filter_personnel_id_str) if filter_personnel_id_str and filter_personnel_id_str.isdigit() else None
                           )
    
@bp.route('/tracking_report')
@login_required
@admin_required
def tracking_report():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))

    # Ambil daftar personel untuk filter dropdown di frontend
    personnel_list = Personnels.query.filter_by(company_obj=company).all()

    # Ambil tanggal filter dari request, default ke hari ini
    filter_date_str = request.args.get('filter_date')
    if filter_date_str:
        try:
            filter_date = datetime.strptime(filter_date_str, "%Y-%m-%d").date()
        except ValueError:
            flash("Invalid date format. Showing report for today.", "warning")
            filter_date = date.today()
    else:
        filter_date = date.today()

    # Logika untuk mendapatkan data tracking report
    # Ini bisa melibatkan Work_Timer (aktivitas duduk, wajah terdeteksi)
    # atau logika yang lebih kompleks dari kamera tracking (Counted_Instances)
    # Untuk contoh ini, saya akan mengambil data Work_Timer yang paling relevan

    tracking_data_query = db.session.query(
        Work_Timer.datetime,
        Work_Timer.type,
        Work_Timer.timer, # Durasi timer (misal, detik)
        Personnels.name.label('employee_name'),
        Camera_Settings.cam_name.label('camera_name')
    ).join(Personnels, Work_Timer.personnel_id == Personnels.id)\
     .join(Camera_Settings, Work_Timer.camera_id == Camera_Settings.id)\
     .filter(
        cast(Work_Timer.datetime, Date) == filter_date,
        Personnels.company_id == company.id,
        Work_Timer.type.in_([Work_Timer.TYPE_SIT, Work_Timer.TYPE_FACE_DETECTED]) # Contoh filter type
    ).order_by(Work_Timer.datetime.asc()).all()

    # Anda mungkin perlu agregasi atau pemrosesan tambahan di sini
    # Misalnya, menghitung total waktu duduk, total waktu wajah terdeteksi,
    # atau log entri/keluar dari area tracking.

    report_data = []
    # Contoh sederhana: hanya menampilkan log individu
    for entry in tracking_data_query:
        report_data.append({
            'datetime': entry.datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'employee_name': entry.employee_name,
            'event_type': entry.type,
            'duration_seconds': entry.timer,
            'camera_name': entry.camera_name
        })

    # Anda bisa memperluas ini untuk agregasi seperti 'work_time_report'
    # For example, to aggregate total sit time or total face detected time per person per day.
    # aggregated_tracking_data = defaultdict(lambda: {'total_sit_time': timedelta(), 'total_face_time': timedelta()})
    # for entry in tracking_data_query:
    #     if entry.type == Work_Timer.TYPE_SIT:
    #         aggregated_tracking_data[entry.employee_name]['total_sit_time'] += timedelta(seconds=entry.timer)
    #     elif entry.type == Work_Timer.TYPE_FACE_DETECTED:
    #         aggregated_tracking_data[entry.employee_name]['total_face_time'] += timedelta(seconds=entry.timer)
    # # Then format aggregated_tracking_data for the template.

    return render_template('admin_panel/tracking_report.html', {
        'personnel_list': personnel_list, # Untuk filter dropdown
        'tracking_report_data': report_data, # Data yang akan ditampilkan
        'filter_date': filter_date.isoformat(), # Tanggal filter
    })

@bp.route('/presence-stream') # URL lebih baik menggunakan tanda hubung
@login_required
# @admin_required # Aktifkan jika perlu
def presence_cam_stream():
    # company = current_user.company # Sesuaikan dengan cara Anda mendapatkan company
    company = Company.query.filter_by(user_id=current_user.id).first()

    if not company:
        flash("User not associated with a company.", "danger")
        # Ganti 'main.index' dengan rute dashboard atau halaman utama Anda
        return redirect(url_for('auth.login')) # Atau halaman lain yang sesuai

    all_active_presence_cameras = Camera_Settings.query.filter_by(
        company_id=company.id, 
        role_camera=Camera_Settings.ROLE_PRESENCE,
        cam_is_active=True
    ).all()  # Asumsi metode query_filter_by ini ada di model Camera_Settings Anda
    
    default_camera = all_active_presence_cameras[0] if all_active_presence_cameras else None
    
    return render_template(
        'admin_panel/presence_cam_stream.html', 
        company=company,
        all_presence_cameras=all_active_presence_cameras,
        default_camera=default_camera,
        hasattr=hasattr, # Pastikan hasattr dikirim jika base template membutuhkannya
        now=datetime.now # Untuk cache buster di URL gambar awal
    )
    
@bp.route('/tracking-stream') # URL lebih baik menggunakan tanda hubung
@login_required
# @admin_required # Aktifkan jika perlu
def tracking_cam_stream():
    # Ambil company berdasarkan user yang login
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login')) # Ganti 'auth.login' dengan rute login Anda

    camera_settings_objects = Camera_Settings.query.filter_by(
        company_id=company.id, 
        role_camera=Camera_Settings.ROLE_TRACKING
    ).all()

    tracking_cameras_list = []
    for cam_obj in camera_settings_objects:
        tracking_cameras_list.append({
            'id': cam_obj.id,
            'cam_name': cam_obj.cam_name,
            'feed_src': cam_obj.feed_src,
            'cam_is_active': cam_obj.cam_is_active 
        })

    return render_template(
        'admin_panel/tracking_cam_stream.html', 
        company=company,
        tracking_cameras=tracking_cameras_list # Pastikan ini adalah list of dicts
    )
def get_relative_image_path(absolute_path):
    if not absolute_path:
        return None
    try:
        # Ini adalah contoh, Anda HARUS menyesuaikannya dengan struktur folder Anda
        # Asumsi UPLOAD_FOLDER adalah 'C:\Users\laila\Desktop\bismillah\coorporate_app\app\static\uploads'
        # dan gambar ada di 'C:\Users\laila\Desktop\bismillah\coorporate_app\app\static\uploads\presence_images\20250529\img.jpg'
        # Maka path relatifnya adalah 'presence_images/20250529/img.jpg' jika UPLOAD_FOLDER adalah basisnya.
        # Atau jika STATIC_FOLDER adalah 'app/static' dan gambar di 'app/static/path/to/image.jpg',
        # maka path relatif adalah 'path/to/image.jpg'
        
        # Cara paling aman adalah menyimpan path RELATIF terhadap folder static di database.
        # Jika Anda menyimpan path absolut, konversinya bisa rumit dan rentan error.
        
        # Contoh paling sederhana jika Anda tahu base path static Anda di server:
        # (Ganti dengan path yang benar di server Anda)
        # static_root_on_server = os.path.normpath("C:/Users/laila/Desktop/bismillah/coorporate_app/app/static")
        # absolute_path_norm = os.path.normpath(absolute_path)

        # if absolute_path_norm.startswith(static_root_on_server):
        #     relative_path = os.path.relpath(absolute_path_norm, static_root_on_server)
        #     return relative_path.replace("\\", "/")

        # Untuk contoh data Anda: C:\Users\laila\Desktop\bismillah\coorporate_app\ap...
        # Ini terlihat seperti ada 'app' di pathnya. Jika folder static Anda adalah 'app/static',
        # dan path di DB adalah C:\Users\laila\Desktop\bismillah\coorporate_app\app\static\folder\gambar.jpg
        # maka Anda perlu mengekstrak 'folder/gambar.jpg'
        
        # Placeholder - Anda perlu implementasi yang benar di sini
        # Misalkan path di DB adalah selalu setelah 'app\static\'
        try:
            base_path_marker = os.path.normpath("app/static/") # Sesuaikan jika perlu
            norm_abs_path = os.path.normpath(absolute_path)
            if base_path_marker in norm_abs_path:
                # Ambil bagian setelah base_path_marker
                rel_path = norm_abs_path.split(base_path_marker, 1)[1]
                return rel_path.replace("\\", "/")
            else:
                # Jika tidak bisa dikonversi, kembalikan None agar template bisa menanganinya
                current_app.logger.warn(f"Tidak bisa mengubah path gambar absolut ke relatif: {absolute_path} menggunakan marker '{base_path_marker}'")
                return None
        except Exception as e_path:
            current_app.logger.error(f"Error saat memproses path gambar '{absolute_path}': {e_path}")
            return None

    except Exception as e:
        current_app.logger.error(f"Error di get_relative_image_path: {e}")
        return None


def get_presences_with_raw_query(company_id_param, filter_date_param_str, filter_personnel_id_param=None):
    current_app.logger.debug(f"MASUK get_presences_with_raw_query: company_id={company_id_param}, date='{filter_date_param_str}', personnel_id='{filter_personnel_id_param}'")

    personnel_entries_table = 'personnel_entries' 
    personnels_table = 'personnels'            
    divisions_table = 'divisions'              

    # Bangun klausa filter personel terlebih dahulu
    personnel_filter_sql_segment = ""
    params = {'filter_date_param': filter_date_param_str, 'company_id_param': company_id_param}

    if filter_personnel_id_param and filter_personnel_id_param.isdigit():
        personnel_filter_sql_segment = f"AND pe.personnel_id = :personnel_id_param" # Filter di CTE DailyEntries
        params['personnel_id_param'] = int(filter_personnel_id_param)

    # Masukkan filter personel ke dalam string query utama menggunakan f-string atau .format
    # sebelum membuat objek text(). Ini lebih aman daripada .replace() pada string query yang kompleks.
    # Pastikan personnel_filter_sql_segment aman (sudah divalidasi isdigit).
    sql_query_str = f"""
    WITH DailyEntries AS (
        SELECT
            pe.personnel_id,
            p.name AS personnel_name,
            p.id AS employee_internal_id,
            pe.timestamp,
            pe.presence_status,
            pe.image AS entry_image_path,
            d.name AS division_name
        FROM
            {personnel_entries_table} pe
        JOIN
            {personnels_table} p ON pe.personnel_id = p.id
        LEFT JOIN
            {divisions_table} d ON p.division_id = d.id
        WHERE
            DATE(pe.timestamp) = :filter_date_param
            AND p.company_id = :company_id_param
            {personnel_filter_sql_segment} -- Placeholder sudah diganti di sini
    ),
    AggregatedPersonnelData AS (
        SELECT
            de.personnel_id,
            de.personnel_name,
            de.employee_internal_id,
            de.division_name,
            MIN(CASE WHEN de.presence_status IN ('ONTIME', 'LATE') THEN de.timestamp END) AS first_in_time,
            MAX(CASE WHEN de.presence_status = 'LEAVE' THEN de.timestamp END) AS last_out_time,
            (SELECT sub_img.image
             FROM {personnel_entries_table} sub_img
             WHERE sub_img.personnel_id = de.personnel_id AND DATE(sub_img.timestamp) = :filter_date_param
             AND sub_img.presence_status IN ('ONTIME', 'LATE')
             ORDER BY sub_img.timestamp ASC LIMIT 1) AS attendance_image,
            (SELECT sub_img_leave.image
             FROM {personnel_entries_table} sub_img_leave
             WHERE sub_img_leave.personnel_id = de.personnel_id AND DATE(sub_img_leave.timestamp) = :filter_date_param
             AND sub_img_leave.presence_status = 'LEAVE'
             ORDER BY sub_img_leave.timestamp DESC LIMIT 1) AS leaving_image,
            (SELECT first_entry.presence_status -- Subquery untuk status check-in pertama
             FROM {personnel_entries_table} first_entry
             WHERE first_entry.personnel_id = de.personnel_id 
             AND first_entry.timestamp = (
                 SELECT MIN(de_sub.timestamp) 
                 FROM DailyEntries de_sub -- Referensi ke CTE DailyEntries di sini mungkin tidak diizinkan di semua DB dalam subquery seperti ini
                                          -- Akan lebih baik jika ini juga diambil dari DailyEntries atau subquery terpisah ke personnel_entries
                 WHERE de_sub.personnel_id = de.personnel_id 
                 AND de_sub.presence_status IN ('ONTIME', 'LATE')
                 -- Pastikan filter tanggal juga ada di subquery ini jika merujuk ke DailyEntries,
                 -- atau jika merujuk langsung ke personnel_entries, tambahkan filter tanggal.
                 -- Untuk amannya, kita referensikan langsung ke personnel_entries dengan filter tanggal
                 -- AND DATE(de_sub.timestamp) = :filter_date_param
             ) 
             -- Jika subquery di atas merujuk ke DailyEntries, mungkin perlu perbaikan.
             -- Mari kita coba referensi ke personnel_entries langsung untuk status ini.
             LIMIT 1 
            ) AS first_check_in_status_val_dummy, -- Ini akan diperbaiki
            (SELECT last_entry.presence_status
             FROM {personnel_entries_table} last_entry
             WHERE last_entry.personnel_id = de.personnel_id AND DATE(last_entry.timestamp) = :filter_date_param
             ORDER BY last_entry.timestamp DESC LIMIT 1
            ) AS overall_last_status_val
        FROM
            DailyEntries de
        GROUP BY
            de.personnel_id, de.personnel_name, de.employee_internal_id, de.division_name
    )
    SELECT
        apd.*,
        (SELECT first_entry.presence_status -- Subquery yang diperbaiki untuk status check-in pertama
         FROM {personnel_entries_table} first_entry
         WHERE first_entry.personnel_id = apd.personnel_id 
         AND first_entry.timestamp = apd.first_in_time -- Gunakan first_in_time yang sudah diagregasi
         LIMIT 1
        ) AS first_check_in_status_val,
        CASE
            WHEN apd.first_in_time IS NOT NULL AND apd.last_out_time IS NOT NULL AND apd.last_out_time > apd.first_in_time
            THEN TIMESTAMPDIFF(SECOND, apd.first_in_time, apd.last_out_time)
            ELSE 0
        END AS total_work_seconds,
        CASE 
            WHEN apd.last_out_time IS NULL AND apd.first_in_time IS NOT NULL THEN 'Masih Bekerja'
            WHEN apd.first_in_time IS NULL THEN 
                (SELECT CASE WHEN COUNT(*) > 0 THEN 'Tidak Ada Check-in Valid (Hanya UNKNOWN)' ELSE 'Tidak Hadir' END 
                 FROM {personnel_entries_table} pe_check -- Cek langsung ke tabel asli
                 WHERE pe_check.personnel_id = apd.personnel_id AND DATE(pe_check.timestamp) = :filter_date_param)
            ELSE 
                CASE 
                    WHEN TIMESTAMPDIFF(SECOND, apd.first_in_time, apd.last_out_time) > (8 * 3600) THEN 'Lembur'
                    WHEN TIMESTAMPDIFF(SECOND, apd.first_in_time, apd.last_out_time) < (7 * 3600) AND apd.last_out_time IS NOT NULL THEN 'Kurang Jam'
                    ELSE 'Standar'
                END
        END AS work_notes_calculated
    FROM AggregatedPersonnelData apd
    ORDER BY apd.personnel_name;
    """
    
    current_app.logger.debug(f"Final SQL Query for Presence:\n{sql_query_str}")
    current_app.logger.debug(f"Params for Presence Query: {params}")

    try:
        result = db.session.execute(text(sql_query_str), params) # Gunakan sql_query_str yang sudah diformat
        entries_raw = result.mappings().all()
        current_app.logger.debug(f"Raw DB Result ({len(entries_raw)} entries): {entries_raw}")

        formatted_presences = []
        for entry in entries_raw:
            work_hours_total_seconds = entry.get('total_work_seconds', 0) if entry.get('total_work_seconds') is not None else 0
            
            work_h = work_hours_total_seconds // 3600
            work_m = (work_hours_total_seconds % 3600) // 60
            
            status_display = entry.get('first_check_in_status_val') 
            notes_val = entry.get('work_notes_calculated', '-')

            if entry.get('overall_last_status_val') == 'LEAVE' and entry.get('last_out_time'):
                status_display = 'PULANG'
            elif notes_val == 'Masih Bekerja':
                status_display = entry.get('first_check_in_status_val') if entry.get('first_check_in_status_val') else 'MASUK'
            elif notes_val == 'Tidak Hadir' or notes_val == 'Tidak Ada Check-in Valid (Hanya UNKNOWN)':
                 status_display = 'TIDAK HADIR'
            elif entry.get('first_check_in_status_val'):
                status_display = entry.get('first_check_in_status_val')
            else: 
                status_display = entry.get('overall_last_status_val', 'N/A')


            attendance_img_rel_path = get_relative_image_path(entry.get('attendance_image'))
            leaving_img_rel_path = get_relative_image_path(entry.get('leaving_image'))

            overtime_hours_str_val = "-"
            if work_hours_total_seconds > (8 * 3600):
                overtime_seconds = work_hours_total_seconds - (8 * 3600)
                ot_h = overtime_seconds // 3600
                ot_m = (overtime_seconds % 3600) // 60
                overtime_hours_str_val = f"{ot_h}j {ot_m}m"

            formatted_presences.append({
                'personnel_id': entry.get('employee_internal_id'),
                'name': entry.get('personnel_name'),
                'attended_time': entry.get('first_in_time'), 
                'attendance_image_path': attendance_img_rel_path,
                'status': status_display,
                'leave_time': entry.get('last_out_time'), 
                'leaving_image_path': leaving_img_rel_path,
                'work_hours_str': f"{work_h}j {work_m}m" if entry.get('first_in_time') and entry.get('last_out_time') else ("-" if not entry.get('first_in_time') else "Belum Pulang"),
                'overtime_hours_str': overtime_hours_str_val if entry.get('first_in_time') and entry.get('last_out_time') else "-",
                'notes': notes_val
            })
        
        current_app.logger.debug(f"Formatted presences untuk dikirim ke template ({len(formatted_presences)}): {formatted_presences}")
        return formatted_presences

    except Exception as e:
        current_app.logger.error(f"Error executing raw query or processing presences: {e}", exc_info=True)
        current_app.logger.error(f"Query yang dieksekusi: {sql_query_str}") # Log query yang sudah diformat
        current_app.logger.error(f"Parameter: {params}")
        flash(f"Terjadi kesalahan Internal Server saat mengambil data absensi.", "danger")
        return []

# Fungsi view utama tetap sama, hanya memanggil get_presences_with_raw_query
@bp.route('/presence-report', methods=['GET'])
@login_required
# @role_required('admin') 
def presence_view():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    personnel_list_for_dropdown = Personnels.query.filter_by(company_id=company.id).order_by(Personnels.name).all()
    filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
    filter_personnel_id = request.args.get('filter_personnel_id')
    
    presence_data_report = get_presences_with_raw_query(company.id, filter_date_str, filter_personnel_id)
    current_app.logger.debug(f"Data yang akan dirender di template (presence_data_report): {presence_data_report}")

    return render_template(
        'admin_panel/presence.html', 
        personnel_list_for_dropdown=personnel_list_for_dropdown,
        presence_data_report=presence_data_report,
        filter_date_str=filter_date_str,
        filter_personnel_id=filter_personnel_id,
        today_date_str=date.today().strftime('%Y-%m-%d'),
        company=company
    )
@bp.route('/presence-report/download', methods=['GET'])
@login_required
def download_presence_excel():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
    filter_personnel_id = request.args.get('filter_personnel_id')

    # Ambil data presensi yang sudah diformat
    presence_data = get_presences_with_raw_query(company.id, filter_date_str, filter_personnel_id)

    # Buat DataFrame dari data presensi
    df = pd.DataFrame(presence_data)

    # Hapus kolom path gambar jika tidak dibutuhkan
    if 'attendance_image_path' in df.columns and 'leaving_image_path' in df.columns:
        df = df.drop(columns=['attendance_image_path', 'leaving_image_path'])

    # Ubah nama kolom untuk Excel agar lebih user-friendly
    df = df.rename(columns={
        'personnel_id': 'ID Karyawan',
        'name': 'Nama',
        'attended_time': 'Waktu Masuk',
        'status': 'Status',
        'leave_time': 'Waktu Pulang',
        'work_hours_str': 'Durasi Kerja',
        'overtime_hours_str': 'Lembur',
        'notes': 'Catatan'
    })

    # Simpan ke Excel dalam memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Presensi')
    output.seek(0)

    filename = f"Laporan_Presensi_{filter_date_str}.xlsx"

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )
