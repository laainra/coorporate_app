from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from flask_login import login_required, current_user, logout_user
from app.models import Company, Divisions, Personnels, Personnel_Entries, Camera_Settings, Work_Timer # Import Work_Timer
from app import db # Import instance db
from app.utils.decorators import admin_required # Decorators
from datetime import datetime, date, timedelta # Import timedelta
from sqlalchemy import func, cast, Date, and_
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
    company = Company.query.filter_by(user_id=current_user).first()
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
    company = Company.query.filter_by(user_id=current_user).first()
    if not company:
        return jsonify({'status': 'error', 'message': 'User does not belong to any company'}), 403
    
    divisions = Divisions.query.filter_by(company_obj=company).all()
    divisions_data = [{'id': div.id, 'name': div.name} for div in divisions]
    return jsonify({'status': 'success', 'divisions': divisions_data})

@bp.route('/add_division', methods=['POST'])
@login_required
@admin_required
def add_division():
    company = Company.query.filter_by(user_id=current_user).first()
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
    company = Company.query.filter_by(user_id=current_user).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))
        
    personnels_list = Personnels.query.filter_by(company_obj=company).all()
    divisions_list = Divisions.query.filter_by(company_obj=company).all()
    return render_template('admin_panel/employees.html', employees=personnels_list, divisions=divisions_list)

@bp.route('/presence')
@login_required
@admin_required
def presence():
    company = Company.query.filter_by(user_id=current_user).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))

    date_str = request.args.get('date')
    personnel_id = request.args.get('personnel_id', type=int)
    
    selected_date = date.today()
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            flash("Invalid date format. Using today's date.", "warning")

    # Filter kehadiran berdasarkan perusahaan admin yang login
    base_query = Personnel_Entries.query.join(Personnels).filter(
        Personnels.company_obj == company,
        cast(Personnel_Entries.timestamp, Date) == selected_date
    )

    if personnel_id:
        base_query = base_query.filter(Personnel_Entries.personnel_id == personnel_id)

    presence_data = base_query.order_by(Personnel_Entries.timestamp.desc()).all()
    personnel_list = Personnels.query.filter_by(company_obj=company).all()

    context = {
        'presence_data': presence_data,
        'personnel_list': personnel_list,
        'selected_date': selected_date.isoformat(), # Format tanggal ke string standar ISO
        'selected_personnel': personnel_id,
        
    }
    return render_template('admin_panel/presence.html', **context, )

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

@bp.route('/work_time_report')
@login_required
@admin_required
def work_time_report():
    company = Company.query.filter_by(user_id=current_user).first()
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

    # Adaptasi Raw SQL Query ke SQLAlchemy ORM atau Hybrid
    # Untuk query yang lebih kompleks dengan join, SQLAlchemy Query Object lebih baik
    # daripada raw SQL string kecuali untuk performa sangat spesifik.
    # Kita akan menggunakan join dan filter SQLAlchemy.

    work_time_query = db.session.query(
        Work_Timer.id,
        Work_Timer.datetime,
        Work_Timer.type,
        Work_Timer.timer,
        Work_Timer.camera_id,
        Work_Timer.personnel_id,
        Personnels.name.label('employee_name'),
        Divisions.name.label('employee_division'),
        Camera_Settings.cam_name.label('camera_name')
    ).join(Personnels, Work_Timer.personnel_id == Personnels.id)\
     .join(Camera_Settings, Work_Timer.camera_id == Camera_Settings.id)\
     .join(Divisions, Personnels.division_id == Divisions.id)\
     .filter(
        cast(Work_Timer.datetime, Date) == filter_date,
        Personnels.company_id == company.id
    ).all() # Ambil semua hasil

    # Agregasi data (menggunakan logika Python/defaultdict yang sama)
    aggregated_data = defaultdict(lambda: {
        'total_time_detected': timedelta(),
        'cctv_areas': set(),
        'employee_id': None,
        'employee_name': None,
        'division': None,
        'date': None,
    })

    for entry in work_time_query:
        timer_seconds = entry.timer
        personnel_id = entry.personnel_id
        employee_name = entry.employee_name
        employee_division = entry.employee_division
        camera_name = entry.camera_name

        total_time = timedelta(seconds=timer_seconds)
        aggregated_data[personnel_id]['total_time_detected'] += total_time
        
        aggregated_data[personnel_id]['employee_id'] = personnel_id
        aggregated_data[personnel_id]['employee_name'] = employee_name
        aggregated_data[personnel_id]['division'] = employee_division
        aggregated_data[personnel_id]['cctv_areas'].add(camera_name)
        aggregated_data[personnel_id]['date'] = entry.datetime.date() # Ambil hanya tanggal dari datetime

    # Siapkan data untuk rendering
    report_data = []
    for data in aggregated_data.values():
        total_seconds = int(data['total_time_detected'].total_seconds())
        total_hours = total_seconds // 3600
        total_minutes = (total_seconds % 3600) // 60
        # total_seconds_remaining = total_seconds % 60 # Jika ingin detik juga

        report_data.append({
            'employee_id': data['employee_id'],
            'employee_name': data['employee_name'],
            'division': data['division'],
            'total_time_hours': total_hours,
            'total_time_minutes': total_minutes,
            'cctv_areas': ', '.join(sorted(list(data['cctv_areas']))), # Join dan sort untuk konsistensi
            'date': data['date'].isoformat(), # Format tanggal ke string standar ISO
        }) 
    
    # Sort report_data by employee name for consistent display
    report_data = sorted(report_data, key=lambda x: x['employee_name'])

    return render_template('admin_panel/work_time_report.html', {
        'personnel_list': personnel_list,
        'work_time_report': report_data,
        'filter_date': filter_date.isoformat(), # Kirim tanggal filter dalam format string ISO
    })
    
    
@bp.route('/tracking_report')
@login_required
@admin_required
def tracking_report():
    company = Company.query.filter_by(user_id=current_user).first()
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
    # company = current_user.company # Sesuaikan
    
    company = Company.query.filter_by(user_id=current_user.id).first()
    if not company:
        flash("User not associated with a company.", "danger")
        return redirect(url_for('auth.login'))
    # Ambil semua kamera pelacak (tracking) yang aktif untuk perusahaan ini
    # Pastikan Anda memiliki Camera_Settings.ROLE_TRACKING atau konstanta yang sesuai
    tracking_cameras = Camera_Settings.query.filter_by(
        company_id=company.id, 
        role_camera=Camera_Settings.ROLE_TRACKING,
        cam_is_active=True # Atau tampilkan semua dan biarkan status diatur di frontend
    )
    
    # Pastikan template HTML Anda bernama 'tracking_cam_stream.html'
    return render_template(
        'admin_panel/tracking_cam_stream.html', 
        company=company,
        tracking_cameras=tracking_cameras, # Kirim daftar kamera ke template
        hasattr=hasattr 
    )
