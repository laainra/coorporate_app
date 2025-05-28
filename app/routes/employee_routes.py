# coorporate_app/app/routes/employee_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user, logout_user
from app.models import Personnels, Personnel_Entries, Personnel_Images # Import models
from app import db # Import instance db
from app.utils.decorators import employee_required # Decorators
from app.utils.config_variables import get_personnel_folder_path
from datetime import datetime, date
from sqlalchemy import func, cast, Date, text # Untuk fungsi database
import os
import shutil # Untuk operasi file/folder
from werkzeug.utils import secure_filename # Untuk upload file

bp = Blueprint('employee', __name__, template_folder='../templates/employee_panel')


def _get_employee_presence_data(target_date=None, personnel_id=None):
    sql = '''
        SELECT 
            p.id AS personnel_id,
            p.name,
            MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END) AS attended_time,
            MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END) AS leaving_time,
            CASE 
                WHEN EXISTS (
                    SELECT 1 
                    FROM personnel_entries AS sub 
                    WHERE sub.personnel_id = p.id 
                    AND sub.presence_status = 'LEAVE'
                ) THEN 'LEAVING'
                ELSE MAX(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.presence_status END)
            END AS latest_status,
            TIMESTAMPDIFF(HOUR, 
                MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
            ) AS work_hours,
            CASE 
                WHEN TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ) > 8 THEN CONCAT('Overtime ', TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ) - 8, ' hours')
                WHEN TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ) < 8 THEN CONCAT('Less time ', 8 - TIMESTAMPDIFF(HOUR, 
                    MIN(CASE WHEN d.presence_status IN ('ONTIME', 'LATE') THEN d.timestamp END),
                    MAX(CASE WHEN d.presence_status = 'LEAVE' THEN d.timestamp END)
                ), ' hours')
                ELSE 'Standard Time'
            END AS notes,
            (SELECT d2.image 
             FROM personnel_entries AS d2 
             WHERE d2.personnel_id = p.id 
             AND d2.presence_status IN ('ONTIME', 'LATE')
             ORDER BY d2.timestamp DESC 
             LIMIT 1
            ) AS attendance_image,
            (SELECT d3.image 
             FROM personnel_entries AS d3 
             WHERE d3.personnel_id = p.id 
             AND d3.presence_status = 'LEAVE'
             ORDER BY d3.timestamp DESC 
             LIMIT 1
            ) AS leaving_image
        FROM 
            personnel_entries AS d
        JOIN 
            personnels AS p ON p.id = d.personnel_id
    '''

    params = {}
    where_clauses = []

    if target_date:
        where_clauses.append("DATE(d.timestamp) = :target_date")
        params['target_date'] = target_date
    if personnel_id:
        where_clauses.append("p.id = :personnel_id")
        params['personnel_id'] = personnel_id

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sql += " GROUP BY p.id"

    result = db.session.execute(text(sql), params)
    entries = result.fetchall()

    presence_data = []
    for entry in entries:
        attended_time = entry.attended_time.strftime('%H:%M:%S') if entry.attended_time else '-'
        leaving_time = entry.leaving_time.strftime('%H:%M:%S') if entry.leaving_time else '-'
        attendance_image_path = f"img/{entry.attendance_image}" if entry.attendance_image else 'img/no_image.png'
        leaving_image_path = f"img/{entry.leaving_image}" if entry.leaving_image else 'img/no_image.png'

        presence_data.append({
            'id': entry.personnel_id,
            'name': entry.name,
            'date': entry.attended_time.date().isoformat() if entry.attended_time else '-',
            'attended': attended_time,
            'leave': leaving_time,
            'status': entry.latest_status,
            'work_hours': entry.work_hours if entry.work_hours is not None else 'Still Working',
            'notes': entry.notes if entry.notes is not None else 'No notes',
            'attendance_image_path': attendance_image_path,
            'leaving_image_path': leaving_image_path,
        })

    return presence_data
@bp.route('/dashboard')
@login_required
@employee_required
def dashboard():
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Your account is not linked to personnel data. Please contact admin.", "danger")
        # Log out user if no personnel data linked, preventing errors
        logout_user() 
        return redirect(url_for('auth.login'))
    
    today = date.today()
    presence_data = _get_employee_presence_data(target_date=today, personnel_id=personnel.id)

    # Anda mungkin ingin mendapatkan ringkasan absensi harian di sini juga
    # Atau ringkasan bulanan untuk chart.
    # Untuk contoh, saya akan membuat dummy stats untuk hari ini.
    
    # Example: Count total presences for today (non-UNKNOWN entries)
    today_total_presences = Personnel_Entries.query.filter(
        Personnel_Entries.personnel_obj == personnel,
        cast(Personnel_Entries.timestamp, Date) == today
    ).filter(Personnel_Entries.presence_status != 'UNKNOWN').count()

    today_ontime = Personnel_Entries.query.filter(
        Personnel_Entries.personnel_obj == personnel,
        cast(Personnel_Entries.timestamp, Date) == today,
        Personnel_Entries.presence_status == 'ONTIME'
    ).count()

    today_late = Personnel_Entries.query.filter(
        Personnel_Entries.personnel_obj == personnel,
        cast(Personnel_Entries.timestamp, Date) == today,
        Personnel_Entries.presence_status == 'LATE'
    ).count()

    today_leave = Personnel_Entries.query.filter(
        Personnel_Entries.personnel_obj == personnel,
        cast(Personnel_Entries.timestamp, Date) == today,
        Personnel_Entries.presence_status == 'LEAVE'
    ).count()

    today_unknown = Personnel_Entries.query.filter(
        Personnel_Entries.personnel_obj == personnel,
        cast(Personnel_Entries.timestamp, Date) == today,
        Personnel_Entries.presence_status == 'UNKNOWN'
    ).count()

    context = {
        'personnel': personnel,
        'presence_data': presence_data, # Data untuk tabel riwayat presensi hari ini
        'today_total_presences': today_total_presences,
        'today_ontime': today_ontime,
        'today_late': today_late,
        'today_leave': today_leave,
        'today_unknown': today_unknown,
        'name': personnel.name
    }
    return render_template('employee_panel/dashboard.html', **context)

@bp.route('/presence_history')
@login_required
@employee_required
def presence_history():
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Your account is not linked to personnel data. Please contact admin.", "danger")
        logout_user()
        return redirect(url_for('auth.login'))

    date_str = request.args.get('date')
    selected_date = None
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            flash("Invalid date format. Showing all history.", "warning")
            selected_date = None

    presence_data = _get_employee_presence_data(target_date=selected_date, personnel_id=personnel.id)

    context = {
        'presence_data': presence_data,
        'selected_date': selected_date.isoformat() if selected_date else None,
        'personnel': personnel # Pass personnel for template display
    }
    return render_template('employee_panel/presence_history.html', **context)

@bp.route('/take_image')
@login_required
@employee_required
def take_image():
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Your account is not linked to personnel data.", "danger")
        return redirect(url_for('employee.dashboard'))
    
    return render_template('employee_panel/capture.html', name=personnel.name, personnel_id=personnel.id)