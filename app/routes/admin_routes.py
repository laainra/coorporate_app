from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from flask_login import login_required, current_user, logout_user
from app.models import Company, Divisions, Personnels, Personnel_Entries, Camera_Settings, Work_Timer, Tracking 
from app import db # Import instance db
from app.utils.decorators import admin_required # Decorators
from datetime import datetime, date, timedelta # Import timedelta
from sqlalchemy import func, cast, Date, and_, text
import io
from PIL import Image
import pandas as pd
from werkzeug.utils import secure_filename # Untuk upload file
import os
import shutil
from collections import defaultdict # Untuk defaultdict
from io import BytesIO

bp = Blueprint('admin', __name__, template_folder='../templates/admin_panel')
# --- Dashboard Route ---
@bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account is not linked to any company. Please contact superadmin.", "danger")
        logout_user()
        return redirect(url_for('auth.login'))

    today = datetime.now().date()

    default_start_date = today - timedelta(days=30)
    default_end_date = today 

    default_start_date_str = default_start_date.strftime('%Y-%m-%d')
    default_end_date_str = default_end_date.strftime('%Y-%m-%d')


    today_summary_query = (
        db.session.query(
            Personnel_Entries.presence_status,
            func.count(Personnel_Entries.id).label('count')
        )
        .join(Personnels, Personnels.id == Personnel_Entries.personnel_id)
        .filter(
            Personnels.company_id == company.id,
            cast(Personnel_Entries.timestamp, Date) == today,
            Personnel_Entries.presence_status.in_(['ONTIME', 'LATE', 'LEAVE', 'OUT_OF_TIME'])
        )
        .group_by(Personnel_Entries.presence_status)
        .all()
    )

    summary_data = {
        'ONTIME': 0,
        'LATE': 0,
        'LEAVE': 0,
        'OUT_OF_TIME': 0
    }
    for row in today_summary_query:
        summary_data[row.presence_status] = row.count

    donut_chart_data = {
        'ontime': summary_data['ONTIME'],
        'late': summary_data['LATE'],
        'leave': summary_data['LEAVE']
    }


    ontime_results = (
        db.session.query(
            Divisions.name.label('division_name'),
            func.count(Personnel_Entries.id).label('count')
        )
        .join(Personnels, Personnels.id == Personnel_Entries.personnel_id)
        .join(Divisions, Divisions.id == Personnels.division_id)
        .filter(
            Personnels.company_id == company.id,
            Personnel_Entries.presence_status == 'ONTIME',
            cast(Personnel_Entries.timestamp, Date) >= default_start_date,
            cast(Personnel_Entries.timestamp, Date) <= default_end_date
        )
        .group_by(Divisions.name)
        .all()
    )

    late_results = (
        db.session.query(
            Divisions.name.label('division_name'),
            func.count(Personnel_Entries.id).label('count')
        )
        .join(Personnels, Personnels.id == Personnel_Entries.personnel_id)
        .join(Divisions, Divisions.id == Personnels.division_id)
        .filter(
            Personnels.company_id == company.id,
            Personnel_Entries.presence_status == 'LATE',
            cast(Personnel_Entries.timestamp, Date) >= default_start_date,
            cast(Personnel_Entries.timestamp, Date) <= default_end_date
        )
        .group_by(Divisions.name)
        .all()
    )

    # Extract all unique division names for categories
    all_division_names = set()
    for row in ontime_results:
        if row.division_name is not None: # Ensure it's not None
            all_division_names.add(str(row.division_name)) # Convert to string
    for row in late_results:
        if row.division_name is not None: # Ensure it's not None
            all_division_names.add(str(row.division_name)) # Convert to string

    categories = sorted(list(all_division_names)) # Sort to ensure consistent order

    # Prepare data for ApexCharts, ensuring default 0 values and correct type
    ontime_data_map = {row.division_name: row.count for row in ontime_results if row.division_name is not None}
    late_data_map = {row.division_name: row.count for row in late_results if row.division_name is not None}

    ontime_series_data = [ontime_data_map.get(cat, 0) for cat in categories]
    late_series_data = [late_data_map.get(cat, 0) for cat in categories]


    bar_chart_data = {
        'categories': categories,
        'series': [
            {
                'name': 'Tepat Waktu',
                'data': ontime_series_data
            },
            {
                'name': 'Telat',
                'data': late_series_data
            }
        ]
    }

    # 3. Top employees data (no changes needed for this error)
    top_employees = (
        db.session.query(
            Personnels.id.label('id'),
            Personnels.name,
            Divisions.name.label('division'),
            func.count(Personnel_Entries.presence_status).label('total_ontime')
        )
        .join(Personnel_Entries, Personnels.id == Personnel_Entries.personnel_id)
        .join(Divisions, Personnels.division_id == Divisions.id)
        .filter(
            Personnels.company_id == company.id,
            Personnel_Entries.presence_status == 'ONTIME'
        )
        .group_by(Personnels.id, Personnels.name, Divisions.name)
        .order_by(func.count(Personnel_Entries.presence_status).desc())
        .limit(5)
        .all()
    )

    return render_template('admin_panel/dashboard.html',
                           company=company,
                           summary=donut_chart_data,
                           presence_data=[],
                           top_employees=top_employees,
                           bar_chart_data=bar_chart_data,
                           default_start_date_str=default_start_date_str,
                           default_end_date_str=default_end_date_str)
    
#  ---- Menampilkan halaman divisi ----
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

# --- Menambahkan divisi baru ---
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
            flash('Division created successfully', 'success')
            return jsonify({'status': 'success', 'message': 'Division created successfully'}) 
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': f'Failed to add division: {str(e)}'}), 500

    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

# --- Mengedit divisi yang sudah ada ---
@bp.route('/edit_division/<int:division_id>', methods=['POST'])
@login_required
@admin_required
def edit_division(division_id):
    division = Divisions.query.get_or_404(division_id)
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

# --- Menghapus divisi yang sudah ada ---
@bp.route('/delete_division/<int:division_id>', methods=['POST'])
@login_required
@admin_required
def delete_division(division_id):
    division = Divisions.query.get_or_404(division_id)
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

# --- Menampilkan halaman karyawan ---
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

#  --- Menampilkan Halaman Kamera Presensi ---
@bp.route('/presence_cam')
@login_required
@admin_required
def presence_cam():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))
    
    cams = Camera_Settings.query.filter(
        Camera_Settings.company_obj == company,
        Camera_Settings.role_camera == Camera_Settings.ROLE_PRESENCE, 

    ).all()
    return render_template(
        'admin_panel/presence_cam.html', 
        cams=cams,
        hasattr=hasattr,    
   
    )

# ---- Menampilkan halaman kamera tracking ----
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

# @bp.route('/work_time_report') 
# @login_required
# def work_time_report():
#     company = Company.query.filter_by(user_id=current_user.id).first()
#     if not company:
#         flash("User tidak terasosiasi dengan perusahaan.", "danger")
#         return redirect(url_for('auth.login'))

#     personnel_list = Personnels.query.filter_by(company_id=company.id).order_by(Personnels.name).all()

#     filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
#     try:
#         filter_date_obj = datetime.strptime(filter_date_str, '%Y-%m-%d').date()
#     except ValueError:
#         flash("Format tanggal tidak valid. Menggunakan tanggal hari ini.", "warning")
#         filter_date_str = date.today().strftime('%Y-%m-%d')
#         filter_date_obj = date.today()

#     filter_personnel_id_str = request.args.get('filter_personnel_id')


#     sql_query = """
#     WITH RankedWorkTimer AS (
#         SELECT 
#             wt.id AS work_timer_id, 
#             wt.datetime AS entry_datetime, 
#             wt.type AS timer_type, 
#             wt.timer AS timer_seconds, 
#             wt.camera_id, 
#             wt.personnel_id,
#             p.name AS employee_name, 
#             d.name AS employee_division, 
#             cs.cam_name AS camera_name,
#             p.id AS employee_internal_id, 
#             -- Jika ada NIP: p.nomor_induk_pegawai AS employee_display_id,
#             ROW_NUMBER() OVER(PARTITION BY wt.personnel_id ORDER BY wt.timer DESC, wt.datetime DESC) as rn
#         FROM 
#             work_timer wt
#         JOIN 
#             personnels p ON wt.personnel_id = p.id
#         JOIN 
#             camera_settings cs ON wt.camera_id = cs.id
#         JOIN 
#             divisions d ON p.division_id = d.id
#         WHERE 
#             DATE(wt.datetime) = :filter_date AND p.company_id = :company_id 
#             {personnel_filter} -- Placeholder untuk filter personel
#     )
#     SELECT 
#         work_timer_id, entry_datetime, timer_type, timer_seconds, camera_id, personnel_id,
#         employee_name, employee_division, camera_name, employee_internal_id
#         -- Jika ada NIP: , employee_display_id
#     FROM RankedWorkTimer
#     WHERE rn = 1
#     ORDER BY employee_name;
#     """
    
#     personnel_filter_sql = ""
#     params = {'filter_date': filter_date_obj, 'company_id': company.id}

#     if filter_personnel_id_str and filter_personnel_id_str.isdigit():
#         personnel_filter_sql = "AND p.id = :personnel_id"
#         params['personnel_id'] = int(filter_personnel_id_str)
    
#     final_sql_query = sql_query.format(personnel_filter=personnel_filter_sql)


#     try:
#         result_proxy = db.session.execute(text(final_sql_query), params)
#         last_timer_entries = [row._asdict() for row in result_proxy]
#     except Exception as e:
#         flash(f"Terjadi error saat mengambil data laporan: {e}", "danger")
#         print(f"SQL Query Error: {e}")
#         print(f"Query: {final_sql_query}")
#         print(f"Params: {params}")
#         last_timer_entries = []

#     report_data_final = []
 
#     for entry_dict in last_timer_entries:
#         total_seconds_val = entry_dict.get('timer_seconds', 0)
#         total_hours = total_seconds_val // 3600
#         remaining_seconds_after_hours = total_seconds_val % 3600
#         total_minutes = remaining_seconds_after_hours // 60
#         final_seconds = remaining_seconds_after_hours % 60
        

#         cctv_area_for_this_entry = entry_dict.get('camera_name', 'N/A')

#         report_data_final.append({

#             'employee_id': entry_dict.get('employee_internal_id', 'N/A'), 
#             'employee_name': entry_dict.get('employee_name', 'N/A'),
#             'division': entry_dict.get('employee_division', 'N/A'),
#             'total_time_hours': total_hours,
#             'total_time_minutes': total_minutes,
#             'total_time_seconds': final_seconds,
#             'cctv_areas': cctv_area_for_this_entry, 
#             'date': entry_dict.get('entry_datetime').date() if entry_dict.get('entry_datetime') else filter_date_obj,
#         })

#     return render_template('admin_panel/work_time_report.html', 
#                            personnel_list=personnel_list,
#                            work_time_report=report_data_final,
#                            filter_date=filter_date_str, 
#                            filter_personnel_id=int(filter_personnel_id_str) if filter_personnel_id_str and filter_personnel_id_str.isdigit() else None
#                            )

@bp.route('/tracking_report')
@login_required
@admin_required
def tracking_report():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))

    # Ambil daftar kamera tracking milik company
    cameras = Camera_Settings.query.filter_by(company_obj=company, role_camera=Camera_Settings.ROLE_TRACKING).all()
    camera_ids = [cam.id for cam in cameras]
    
    # Ambil daftar personel untuk dropdown
    personnels_list = Personnels.query.filter_by(company_obj=company).all()

    # Ambil tanggal filter dari request, jika tidak ada tampilkan semua data
    filter_date_str = request.args.get('filter_date')
    filter_personnel_id = request.args.get('filter_personnel_id')
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Jumlah data per halaman
    tracking_query = Tracking.query.join(Camera_Settings).filter(
        Tracking.camera_id.in_(camera_ids)
    )
    if filter_date_str:
        try:
            filter_date = datetime.strptime(filter_date_str, "%Y-%m-%d").date()
            tracking_query = tracking_query.filter(db.func.date(Tracking.timestamp) == filter_date)
        except ValueError:
            flash("Invalid date format. Showing all data.", "warning")
            filter_date = None
    else:
        filter_date = None  # Tidak ada filter, tampilkan semua data
        
    # Filter by personnel jika ada
    if filter_personnel_id and filter_personnel_id.isdigit():
        tracking_query = tracking_query.filter(Tracking.personnel_id == int(filter_personnel_id))

    tracking_query = tracking_query.order_by(Tracking.timestamp.desc())
    pagination = tracking_query.paginate(page=page, per_page=per_page, error_out=False)
    tracking_entries = pagination.items

    tracking_data = []
    for idx, entry in enumerate(tracking_entries, start=1 + (page-1)*per_page):
        image_url = None
        if entry.image_path:
            if entry.image_path.startswith('static/'):
                image_url = url_for('static', filename=entry.image_path[7:])
            else:
                image_url = url_for('static', filename=entry.image_path)
        personnel_name = entry.personnel.name if entry.personnel else '-'
        tracking_data.append({
            'no': idx,
            'tracking_id': entry.id,  # <-- Tambahkan ini
            'timestamp': entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'detected_class': entry.detected_class,
            'confidence': entry.confidence,
            'image_url': image_url,
            'personnel_name': personnel_name,  # <-- Tambahkan ini
            'camera_name': entry.camera.cam_name if entry.camera else '-',
        })

    return render_template(
        'admin_panel/tracking_report.html',
        tracking_data=tracking_data,
        filter_date=filter_date_str if filter_date_str else '',
        filter_personnel_id=filter_personnel_id,
        personnels_list=personnels_list,
        cameras=cameras,
        pagination=pagination
    )

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
def get_relative_image_path(db_image_path):
    """
    Memastikan path gambar dari database valid dan siap digunakan di template.
    Jika path sudah relatif terhadap folder static, kita bisa langsung menggunakannya.
    """
    if not db_image_path:
        return None
    
    # Asumsi db_image_path sudah merupakan path relatif yang benar dari folder static
    # Misalnya: 'img/extracted_faces/predicted_faces/absence/20250604/file.jpg'
    # Tidak perlu konversi lebih lanjut jika sudah demikian.
    # Anda mungkin ingin menambahkan validasi apakah file tersebut benar-benar ada di server
    # sebelum mengembalikannya, tapi itu opsional dan bisa menambah overhead.

    # Pastikan menggunakan forward slashes untuk URL web
    return str(db_image_path).replace("\\", "/")

# Your existing function, modified
def get_presences_with_raw_query(company_id_param, filter_date_param_str, filter_personnel_id_param=None):
    current_app.logger.debug(f"MASUK get_presences_with_raw_query: company_id={company_id_param}, date='{filter_date_param_str}', personnel_id='{filter_personnel_id_param}'")

    personnel_entries_table = 'personnel_entries'
    personnels_table = 'personnels'
    divisions_table = 'divisions'

    # Bangun klausa filter personel
    personnel_where_clause = ""
    params = {
        'filter_date_param': filter_date_param_str,
        'company_id_param': company_id_param
    }

    if filter_personnel_id_param and filter_personnel_id_param.isdigit():
        personnel_where_clause = f"WHERE p.id = :personnel_id_param"
        params['personnel_id_param'] = int(filter_personnel_id_param)

    sql_query_str = f"""
    WITH DailyEntries AS (
        SELECT
            pe.personnel_id,
            pe.timestamp,
            pe.presence_status,
            pe.image
        FROM
            {personnel_entries_table} pe
        WHERE
            DATE(pe.timestamp) = :filter_date_param
    ),
    AggregatedDailyEntries AS (
        SELECT
            de.personnel_id,
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
            (SELECT first_entry.presence_status
             FROM {personnel_entries_table} first_entry
             WHERE first_entry.personnel_id = de.personnel_id
             AND first_entry.timestamp = (
                 SELECT MIN(sub_de.timestamp)
                 FROM DailyEntries sub_de
                 WHERE sub_de.personnel_id = de.personnel_id
                 AND sub_de.presence_status IN ('ONTIME', 'LATE')
             ) LIMIT 1
            ) AS first_check_in_status_val,
            (SELECT last_entry.presence_status
             FROM {personnel_entries_table} last_entry
             WHERE last_entry.personnel_id = de.personnel_id AND DATE(last_entry.timestamp) = :filter_date_param
             ORDER BY last_entry.timestamp DESC LIMIT 1
            ) AS overall_last_status_val
        FROM
            DailyEntries de
        GROUP BY
            de.personnel_id
    )
    SELECT
        p.id AS personnel_id,
        p.name AS personnel_name,
        p.id AS employee_internal_id, -- Assuming p.id is your internal employee ID
        d.name AS division_name,
        ade.first_in_time,
        ade.last_out_time,
        ade.attendance_image,
        ade.leaving_image,
        ade.first_check_in_status_val,
        ade.overall_last_status_val,
        CASE
            WHEN ade.first_in_time IS NOT NULL AND ade.last_out_time IS NOT NULL AND ade.last_out_time > ade.first_in_time
            THEN TIMESTAMPDIFF(SECOND, ade.first_in_time, ade.last_out_time)
            ELSE 0
        END AS total_work_seconds
    FROM
        {personnels_table} p
    LEFT JOIN
        {divisions_table} d ON p.division_id = d.id
    LEFT JOIN
        AggregatedDailyEntries ade ON p.id = ade.personnel_id
    WHERE
        p.company_id = :company_id_param
        {personnel_where_clause}
    ORDER BY
        p.name;
    """

    current_app.logger.debug(f"Final SQL Query for Presence:\n{sql_query_str}")
    current_app.logger.debug(f"Params for Presence Query: {params}")

    try:
        result = db.session.execute(text(sql_query_str), params)
        entries_raw = result.mappings().all()
        current_app.logger.debug(f"Raw DB Result ({len(entries_raw)} entries): {entries_raw}")

        formatted_presences = []
        for entry in entries_raw:
            # Initialize with default "belum absen" values
            status_display = 'BELUM ABSEN'
            attended_time_str = '-'
            attendance_img_rel_path = None
            leave_time_str = '-'
            leaving_img_rel_path = None
            work_hours_str = '-'
            overtime_hours_str_val = '-'
            notes_val = '-' # Default notes for 'BELUM ABSEN'

            first_in_time_dt = entry.get('first_in_time')
            last_out_time_dt = entry.get('last_out_time')
            first_check_in_status = entry.get('first_check_in_status_val')
            overall_last_status = entry.get('overall_last_status_val')
            work_hours_total_seconds = entry.get('total_work_seconds', 0) # Default to 0 if null

            if first_in_time_dt:
                attended_time_str = first_in_time_dt.strftime('%H:%M:%S')
                attendance_img_rel_path = get_relative_image_path(entry.get('attendance_image'))

                # Determine initial status
                if first_check_in_status == 'ONTIME':
                    status_display = 'ONTIME'
                elif first_check_in_status == 'LATE':
                    status_display = 'LATE'
                # For 'OUT_OF_ATTENDANCE_TIME' at initial check-in, you might display it directly
                # or categorize it as a specific 'Tidak Hadir' type.
                # For simplicity, if it's the first check-in and it's not ONTIME/LATE, we can call it 'OUT_OF_ATTENDANCE_TIME'
                # if first_check_in_status == 'OUT_OF_ATTENDANCE_TIME':
                #     status_display = 'OUT OF ATTENDANCE TIME'

                # Calculate work hours if both in and out times exist
                if last_out_time_dt:
                    leave_time_str = last_out_time_dt.strftime('%H:%M:%S')
                    leaving_img_rel_path = get_relative_image_path(entry.get('leaving_image'))

                    work_h = work_hours_total_seconds // 3600
                    work_m = (work_hours_total_seconds % 3600) // 60
                    work_hours_str = f"{work_h}j {work_m}m"

                    # Determine notes based on calculated work hours
                    if work_hours_total_seconds > (8 * 3600):
                        overtime_seconds = work_hours_total_seconds - (8 * 3600)
                        ot_h = overtime_seconds // 3600
                        ot_m = (overtime_seconds % 3600) // 60
                        overtime_hours_str_val = f"{ot_h}j {ot_m}m"
                        notes_val = 'Lembur'
                    elif work_hours_total_seconds < (7 * 3600): # Assuming 7 hours as minimum for 'Kurang Jam'
                        notes_val = 'Kurang Jam'
                    else:
                        notes_val = 'Standar'
                    
                    # Override status if employee has left
                    if overall_last_status == 'LEAVE':
                        status_display = 'PULANG'
                else: # first_in_time exists, but last_out_time is None (still working)
                    work_hours_str = "Belum Pulang"
                    notes_val = 'Masih Bekerja'
                    # If they checked in, the status is their first check-in status (ONTIME/LATE)
                    status_display = first_check_in_status if first_check_in_status else 'MASUK'


            # If no first_in_time, then they haven't attended yet
            if not first_in_time_dt:
                status_display = 'BELUM ABSEN'
                attended_time_str = '-'
                attendance_img_rel_path = None
                leave_time_str = '-'
                leaving_img_rel_path = None
                work_hours_str = '-'
                overtime_hours_str_val = '-'
                notes_val = '-'


            formatted_presences.append({
                'personnel_id': entry.get('employee_internal_id'),
                'name': entry.get('personnel_name'),
                'attended_time': attended_time_str,
                'attendance_image_path': attendance_img_rel_path,
                'status': status_display,
                'leave_time': leave_time_str,
                'leaving_image_path': leaving_img_rel_path,
                'work_hours_str': work_hours_str,
                'overtime_hours_str': overtime_hours_str_val,
                'notes': notes_val
            })

        current_app.logger.debug(f"Formatted presences untuk dikirim ke template ({len(formatted_presences)}): {formatted_presences}")
        return formatted_presences

    except Exception as e:
        current_app.logger.error(f"Error executing raw query or processing presences: {e}", exc_info=True)
        current_app.logger.error(f"Query yang dieksekusi: {sql_query_str}")
        current_app.logger.error(f"Parameter: {params}")
        # flash(f"Terjadi kesalahan Internal Server saat mengambil data absensi.", "danger") # Uncomment in Flask app
        return []
    
@bp.route('/presence-report', methods=['GET'])
@login_required
def presence_view():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    personnel_list_for_dropdown = Personnels.query.filter_by(company_id=company.id).order_by(Personnels.name).all()
    filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
    filter_personnel_id = request.args.get('filter_personnel_id')

    all_presence_data = get_presences_with_raw_query(company.id, filter_date_str, filter_personnel_id)


    page = request.args.get('page', 1, type=int) 
    per_page = 10


    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_presence_data = all_presence_data[start_index:end_index]

    total_items = len(all_presence_data)
    total_pages = (total_items + per_page - 1) // per_page 
    

    class Pagination:
        def __init__(self, page, per_page, total_count, items):
            self.page = page
            self.per_page = per_page
            self.total = total_count
            self.items = items
            self.pages = (total_count + per_page - 1) // per_page
            self.has_prev = self.page > 1
            self.has_next = self.page < self.pages
            self.prev_num = self.page - 1
            self.next_num = self.page + 1

        def iter_pages(self, left_edge=2, right_edge=2, left_current=2, right_current=2):
            last = 0
            for num in range(1, self.pages + 1):
                if num <= left_edge or \
                   (num > self.page - left_current - 1 and num < self.page + right_current + 1) or \
                   num > self.pages - right_edge:
                    if last + 1 != num:
                        yield None
                    yield num
                    last = num

    presence_pagination = Pagination(page, per_page, total_items, paginated_presence_data)


    current_app.logger.debug(f"Data yang akan dirender di template (paginated_presence_data): {len(paginated_presence_data)} entries")

    return render_template(
        'admin_panel/presence.html',
        personnel_list_for_dropdown=personnel_list_for_dropdown,
        presence_data_report=paginated_presence_data, # Kirim data yang sudah dipaginasi
        pagination=presence_pagination, # Kirim objek pagination
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

    presence_data_report = get_presences_with_raw_query(company.id, filter_date_str, filter_personnel_id)

    # Convert to DataFrame
    df = pd.DataFrame(presence_data_report)

    # Rename columns for nicer headers
    df.rename(columns={
        'personnel_id': 'ID Pegawai',
        'name': 'Nama',
        'attended_time': 'Waktu Masuk',
        'attendance_image_path': 'Foto Masuk',
        'status': 'Status',
        'leave_time': 'Waktu Keluar',
        'leaving_image_path': 'Foto Pulang',
        'work_hours_str': 'Total Jam Kerja',
        'overtime_hours_str': 'Lembur',
        'notes': 'Catatan'
    }, inplace=True)

    # Export to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Presensi')

    output.seek(0)
    filename = f"Laporan_Presensi_{filter_date_str}.xlsx"
    return send_file(output, download_name=filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# @bp.route('/export_work_time_report_excel')
# @login_required
# def export_work_time_report_excel():
#     company = Company.query.filter_by(user_id=current_user.id).first()
#     if not company:
#         flash("User tidak terasosiasi dengan perusahaan.", "danger")
#         return redirect(url_for('auth.login'))

#     filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
#     try:
#         filter_date_obj = datetime.strptime(filter_date_str, '%Y-%m-%d').date()
#     except ValueError:
#         filter_date_obj = date.today()

#     filter_personnel_id_str = request.args.get('filter_personnel_id')

#     sql_query = """
#     WITH RankedWorkTimer AS (
#         SELECT 
#             wt.id AS work_timer_id, 
#             wt.datetime AS entry_datetime, 
#             wt.type AS timer_type, 
#             wt.timer AS timer_seconds, 
#             wt.camera_id, 
#             wt.personnel_id,
#             p.name AS employee_name, 
#             d.name AS employee_division, 
#             cs.cam_name AS camera_name,
#             p.id AS employee_internal_id,
#             ROW_NUMBER() OVER(PARTITION BY wt.personnel_id ORDER BY wt.timer DESC, wt.datetime DESC) as rn
#         FROM 
#             work_timer wt
#         JOIN 
#             personnels p ON wt.personnel_id = p.id
#         JOIN 
#             camera_settings cs ON wt.camera_id = cs.id
#         JOIN 
#             divisions d ON p.division_id = d.id
#         WHERE 
#             DATE(wt.datetime) = :filter_date AND p.company_id = :company_id 
#             {personnel_filter}
#     )
#     SELECT 
#         work_timer_id, entry_datetime, timer_type, timer_seconds, camera_id, personnel_id,
#         employee_name, employee_division, camera_name, employee_internal_id
#     FROM RankedWorkTimer
#     WHERE rn = 1
#     ORDER BY employee_name;
#     """

#     personnel_filter_sql = ""
#     params = {'filter_date': filter_date_obj, 'company_id': company.id}

#     if filter_personnel_id_str and filter_personnel_id_str.isdigit():
#         personnel_filter_sql = "AND p.id = :personnel_id"
#         params['personnel_id'] = int(filter_personnel_id_str)

#     final_sql_query = sql_query.format(personnel_filter=personnel_filter_sql)

#     try:
#         result_proxy = db.session.execute(text(final_sql_query), params)
#         last_timer_entries = [row._asdict() for row in result_proxy]
#     except Exception as e:
#         flash(f"Terjadi error saat mengekspor laporan: {e}", "danger")
#         return redirect(url_for('admin.work_time_report'))

#     report_data_final = []

#     for entry in last_timer_entries:
#         seconds = entry.get('timer_seconds', 0)
#         hours = seconds // 3600
#         minutes = (seconds % 3600) // 60
#         sec = seconds % 60

#         report_data_final.append({
#             'ID Pegawai': entry.get('employee_internal_id', 'N/A'),
#             'Nama Pegawai': entry.get('employee_name', 'N/A'),
#             'Divisi': entry.get('employee_division', 'N/A'),
#             'Jam': hours,
#             'Menit': minutes,
#             'Detik': sec,
#             'Area CCTV': entry.get('camera_name', 'N/A'),
#             'Tanggal': entry.get('entry_datetime').strftime('%Y-%m-%d') if entry.get('entry_datetime') else str(filter_date_obj),
#         })

#     # Convert ke DataFrame
#     df = pd.DataFrame(report_data_final)

#     # Simpan ke Excel di memory (stream)
#     output = io.BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False, sheet_name='Laporan Waktu Kerja')

#     output.seek(0)
#     filename = f"Laporan_Waktu_Kerja_{filter_date_obj}.xlsx"
#     return send_file(output,
#                      as_attachment=True,
#                      download_name=filename,
#                      mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def _get_work_time_data(filter_date_obj, filter_personnel_id_str, company_id):
    """
    Fungsi helper untuk mengambil data laporan waktu kerja dari database.
    Mencegah duplikasi kode antara route web dan route export.
    """
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
    FROM RankedWorkTimer
    WHERE rn = 1
    ORDER BY employee_name;
    """
    
    personnel_filter_sql = ""
    params = {'filter_date': filter_date_obj, 'company_id': company_id}

    if filter_personnel_id_str and filter_personnel_id_str.isdigit():
        personnel_filter_sql = "AND p.id = :personnel_id"
        params['personnel_id'] = int(filter_personnel_id_str)
    
    final_sql_query = sql_query.format(personnel_filter=personnel_filter_sql)

    try:
        result_proxy = db.session.execute(text(final_sql_query), params)
        return [row._asdict() for row in result_proxy]
    except Exception as e:
        flash(f"Terjadi error saat mengambil data laporan: {e}", "danger")
        print(f"SQL Query Error: {e}\nQuery: {final_sql_query}\nParams: {params}")
        return []


@bp.route('/work-time-report')
@login_required
def work_time_report():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    personnel_list = Personnels.query.filter_by(company_id=company.id).order_by(Personnels.name).all()

    # Ambil filter dari request.args (GET)
    filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
    filter_personnel_id_str = request.args.get('filter_personnel_id')

    try:
        filter_date_obj = datetime.strptime(filter_date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        flash("Format tanggal tidak valid. Menggunakan tanggal hari ini.", "warning")
        filter_date_str = date.today().strftime('%Y-%m-%d')
        filter_date_obj = date.today()

    # Panggil fungsi helper untuk mendapatkan data
    fetched_entries = _get_work_time_data(filter_date_obj, filter_personnel_id_str, company.id)
    
    report_data_final = []
    for entry_dict in fetched_entries:
        total_seconds_val = entry_dict.get('timer_seconds', 0)
        total_hours = total_seconds_val // 3600
        remaining_seconds_after_hours = total_seconds_val % 3600
        total_minutes = remaining_seconds_after_hours // 60
        final_seconds = remaining_seconds_after_hours % 60
        
        report_data_final.append({
            'employee_id': entry_dict.get('employee_internal_id', 'N/A'),
            'employee_name': entry_dict.get('employee_name', 'N/A'),
            'division': entry_dict.get('employee_division', 'N/A'),
            'total_time_hours': total_hours,
            'total_time_minutes': total_minutes,
            'total_time_seconds': final_seconds,
            'cctv_areas': entry_dict.get('camera_name', 'N/A'),
            'date': entry_dict.get('entry_datetime').date() if entry_dict.get('entry_datetime') else filter_date_obj,
        })

    return render_template('admin_panel/work_time_report.html', 
                           personnel_list=personnel_list,
                           work_time_report=report_data_final,
                           filter_date=filter_date_str, 
                           filter_personnel_id=int(filter_personnel_id_str) if filter_personnel_id_str and filter_personnel_id_str.isdigit() else None
                           )


# Di file Python Anda (routes.py atau sejenisnya)
@bp.route('/export_work_time_report_excel')
@login_required
def export_work_time_report_excel():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    filter_date_str = request.args.get('filter_date', date.today().strftime('%Y-%m-%d'))
    filter_personnel_id_str = request.args.get('filter_personnel_id')

    try:
        filter_date_obj = datetime.strptime(filter_date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        filter_date_obj = date.today()

    fetched_entries = _get_work_time_data(filter_date_obj, filter_personnel_id_str, company.id)

    report_data_final = []
    if fetched_entries:
        for entry in fetched_entries:
            seconds = entry.get('timer_seconds', 0)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            sec = seconds % 60
            report_data_final.append({
                'Tanggal': entry.get('entry_datetime').strftime('%Y-%m-%d') if entry.get('entry_datetime') else str(filter_date_obj),
                'ID Karyawan': entry.get('employee_internal_id', 'N/A'),
                'Nama Karyawan': entry.get('employee_name', 'N/A'),
                'Divisi': entry.get('employee_division', 'N/A'),
                'Jam': hours, 'Menit': minutes, 'Detik': sec,
                'Area CCTV': entry.get('camera_name', 'N/A'),
            })

    df = pd.DataFrame(report_data_final)
    
    # Jika DataFrame kosong, buat ulang dengan header yang benar
    if df.empty:
        df = pd.DataFrame(columns=[
            'Tanggal', 'ID Karyawan', 'Nama Karyawan', 'Divisi', 
            'Jam', 'Menit', 'Detik', 'Area CCTV'
        ])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Laporan Waktu Kerja')
    output.seek(0)
    
    filename = f"Laporan_Waktu_Kerja_{filter_date_obj}.xlsx"
    return send_file(output,
                     as_attachment=True,
                     download_name=filename,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    

@bp.route('/export_tracking_report_excel')
@login_required
def export_tracking_report_excel():
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User tidak terasosiasi dengan perusahaan.", "danger")
        return redirect(url_for('auth.login'))

    filter_date_str = request.args.get('filter_date')
    cameras = Camera_Settings.query.filter_by(company_obj=company, role_camera=Camera_Settings.ROLE_TRACKING).all()
    camera_ids = [cam.id for cam in cameras]

    tracking_query = Tracking.query.join(Camera_Settings).filter(
        Tracking.camera_id.in_(camera_ids)
    )
    if filter_date_str:
        try:
            filter_date = datetime.strptime(filter_date_str, "%Y-%m-%d").date()
            tracking_query = tracking_query.filter(db.func.date(Tracking.timestamp) == filter_date)
        except ValueError:
            filter_date = None

    tracking_query = tracking_query.order_by(Tracking.timestamp.desc())

    report_data = []
    for idx, entry in enumerate(tracking_query, start=1):
        if entry.detected_class == 'person_no_tie':
            pelanggaran = 'Tidak memakai dasi'
        else:
            pelanggaran = entry.detected_class

        image_display = entry.image_path if entry.image_path else '-'
        personnel_name = entry.personnel.name if entry.personnel else '-'
        report_data.append({
            'No.': idx,
            'Tanggal': entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Object Pelanggaran': pelanggaran,
            'Confidence': "%.2f" % entry.confidence,
            'Captured Image': image_display,
            'Nama Pegawai': personnel_name,  # <-- Tambahkan ini
            'Area CCTV': entry.camera.cam_name if entry.camera else '-',
        })

    df = pd.DataFrame(report_data)
    if df.empty:
        df = pd.DataFrame(columns=[
            'No.', 'Tanggal', 'Object Pelanggaran', 'Confidence',
            'Captured Image', 'Nama Pegawai', 'Area CCTV'
        ])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Hapus 'Captured Image' sebelum menulis ke Excel, kita akan menanganinya secara manual
        df_for_excel = df.drop(columns=['Captured Image'])
        df_for_excel.insert(4, 'Captured Image', None) # Sisipkan kolom kosong untuk gambar
        
        df_for_excel.to_excel(writer, index=False, sheet_name='Tracking Report', startrow=1, header=False)

        workbook  = writer.book
        worksheet = writer.sheets['Tracking Report']

        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'vcenter',
            'align': 'center',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # Tulis header secara manual
        headers = list(df.columns)
        for col_num, value in enumerate(headers):
            worksheet.write(0, col_num, value, header_format)

        # Atur lebar kolom
        for i, col in enumerate(headers):
            if col == 'Captured Image':
                worksheet.set_column(i, i, 30)
            else:
                column_len = max(df[col].astype(str).map(len).max(), len(col))
                worksheet.set_column(i, i, column_len + 2)

        img_col_idx = headers.index('Captured Image')

        # Set isi cell data ke tengah
        n_rows, n_cols = df_for_excel.shape
        for row in range(1, n_rows + 1):  # Mulai dari 1 karena header di 0
            for col in range(n_cols):
                worksheet.write(row, col, df_for_excel.iloc[row-1, col], center_format)

        # Sisipkan gambar (seperti sebelumnya)
        image_row_height = 95
        for row_num, img_path in enumerate(df['Captured Image'], start=2):
            worksheet.set_row(row_num - 1, image_row_height)
            if img_path and img_path != '-':
                img_path_abs = os.path.join(current_app.root_path, img_path)
                if os.path.exists(img_path_abs):
                    try:
                        with Image.open(img_path_abs) as img:
                            img_width, img_height = img.size
                        scale = image_row_height / img_height
                        worksheet.insert_image(
                            row_num - 1, img_col_idx, img_path_abs,
                            {
                                'x_scale': scale,
                                'y_scale': scale,
                                'object_position': 1,
                                'x_offset': 5,
                                'y_offset': 5
                            }
                        )
                    except Exception as e:
                        worksheet.write(row_num - 1, img_col_idx, f"Error: {e}", center_format)
                else:
                    worksheet.write(row_num - 1, img_col_idx, 'File not found', center_format)
    
    output.seek(0)
    filename = f"Tracking_Report_{filter_date_str if filter_date_str else 'all'}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@bp.route('/delete_tracking/<int:tracking_id>', methods=['POST'])
@login_required
@admin_required
def delete_tracking(tracking_id):
    tracking_entry = Tracking.query.get_or_404(tracking_id)
    try:
        db.session.delete(tracking_entry)
        db.session.commit()
        flash('Data tracking berhasil dihapus.', 'success')
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Gagal menghapus data: {e}'}), 500
