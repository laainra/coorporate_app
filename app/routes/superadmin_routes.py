# coorporate_app/app/routes/superadmin_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models import Company, Users, Personnels # Import models
from app import db # Import instance db
from app.utils.decorators import superadmin_required # Decorators
from datetime import datetime, timedelta
from sqlalchemy import func, cast, Date # Untuk fungsi database
import pandas as pd # Untuk manipulasi data dan charting
bp = Blueprint('superadmin', __name__, template_folder='../templates/superadmin')
@bp.route('/dashboard')
@login_required
@superadmin_required
def dashboard():
    today = datetime.utcnow()
    last_7_days_ago = today - timedelta(days=7)
    last_30_days_ago = today - timedelta(days=30)

    total_companies = Company.query.count()
    companies_last_7_days = Company.query.filter(Company.createdAt >= last_7_days_ago).count()
    companies_last_30_days = Company.query.filter(Company.createdAt >= last_30_days_ago).count()

    total_accounts = Users.query.count()
    accounts_last_7_days = Users.query.filter(Users.createdAt >= last_7_days_ago).count()
    accounts_last_30_days = Users.query.filter(Users.createdAt >= last_30_days_ago).count()

    total_employees = Personnels.query.count()
    employees_last_7_days = Personnels.query.filter(Personnels.createdAt >= last_7_days_ago).count()
    employees_last_30_days = Personnels.query.filter(Personnels.createdAt >= last_30_days_ago).count()

    start_date = today - timedelta(days=29)

    def get_daily_cumulative_data(model, start_date):
        daily_counts = db.session.query(
            func.date(model.createdAt).label('date'),
            func.count(model.id).label('count')
        ).filter(model.createdAt >= start_date).group_by('date').order_by('date').all()
        
        if daily_counts:
            df = pd.DataFrame(daily_counts, columns=['date', 'count'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            full_date_range = pd.date_range(start=start_date.date(), end=today.date(), freq='D')
            df = df.reindex(full_date_range, fill_value=0)

            cumulative_series = df['count'].cumsum()
            return cumulative_series
        return pd.Series()

    company_growth = get_daily_cumulative_data(Company, start_date)
    user_growth = get_daily_cumulative_data(Users, start_date)
    employee_growth = get_daily_cumulative_data(Personnels, start_date)


    chart_data = {
        # Format label tanggal menjadi "Bulan Hari", contoh: "Jun 09"
        'labels': [d.strftime('%b %d') for d in company_growth.index],
        'company_series': company_growth.tolist(),
        'user_series': user_growth.tolist(),
        'employee_series': employee_growth.tolist()
    }
    
    # Kumpulkan semua data untuk dikirim ke template
    context = {
        'total_companies': total_companies,
        'companies_last_7_days': companies_last_7_days,
        'companies_last_30_days': companies_last_30_days,
        'total_accounts': total_accounts,
        'accounts_last_7_days': accounts_last_7_days,
        'accounts_last_30_days': accounts_last_30_days,
        'total_employees': total_employees,
        'employees_last_7_days': employees_last_7_days,
        'employees_last_30_days': employees_last_30_days,
        'chart_data': chart_data  # Tambahkan data chart ke context
    }
    
    return render_template('superadmin/dashboard.html', **context)

@bp.route('/company')
@login_required
@superadmin_required
def company():
    search_term = request.args.get('search', '').lower()
    entries_per_page = int(request.args.get('entries', 10))
    page = int(request.args.get('page', 1))

    companies_query = Company.query
    if search_term:
        companies_query = companies_query.filter(Company.name.ilike(f'%{search_term}%')) 

    total_count = companies_query.count()
    companies = companies_query.offset((page - 1) * entries_per_page).limit(entries_per_page).all()
    

    total_pages = (total_count + entries_per_page - 1) // entries_per_page

    context = {
        'companies': companies,
        'company_count': total_count,
        'entries_per_page': entries_per_page,
        'search_term': search_term,
        'page': page,
        'total_pages': total_pages,
    }
    return render_template('superadmin/company_list.html', **context)


@bp.route('/company/<int:company_id>')
@login_required
@superadmin_required
def get_company(company_id):
    company = Company.query.get_or_404(company_id)
    # Pastikan user account terkait ada
    username = company.user_account.username if company.user_account else None
    email = company.user_account.email if company.user_account else None
    
    # Password tidak boleh dikirim langsung. Ini hanya placeholder.
    # Jangan pernah mengirim password plainteks atau hash password melalui JSON untuk UI.
    # Jika perlu password untuk edit, Anda harus meminta input password baru.
    password_dummy = "********" 

    data = {
        'id': company_id,
        'company_name': company.name,
        'username': username,
        'email': email,
        'password': password_dummy, # Placeholder
    }
    return jsonify(data)

@bp.route('/company/add', methods=['POST'])
@login_required
@superadmin_required
def add_company():
    if request.method == "POST":
        company_name = request.form.get('company_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([company_name, username, password]):
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400

        if Users.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists.'}), 409
        if email and Users.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already exists.'}), 409

        try:
            new_user = Users(username=username, email=email, role=Users.ROLE_ADMIN)
            new_user.set_password(password) 
            db.session.add(new_user)
            db.session.flush() 

            new_company = Company(name=company_name, user_id=new_user.id)
            db.session.add(new_company)
            db.session.commit()
            flash('Company and admin user created successfully!', 'success')
            return jsonify({'success': True, 'message': 'Company created successfully!'})
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating company: {str(e)}', 'danger')
            return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': False, 'message': 'Method Not Allowed'}), 405

@bp.route('/company/edit/<int:company_id>', methods=['POST'])
@login_required
@superadmin_required
def edit_company(company_id):
    if request.method == "POST":
        company = Company.query.get_or_404(company_id)
        company_name = request.form.get('company_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password') 

        try:
            company.name = company_name if company_name else company.name

            user_account = company.user_account 
            if user_account:
                user_account.username = username if username else user_account.username
                user_account.email = email if email else user_account.email
                if password:
                    user_account.set_password(password)
                db.session.add(user_account) 
            
            db.session.add(company) 
            db.session.commit()
            flash('Company updated successfully!', 'success')
            return jsonify({'success': True, 'message': 'Company updated successfully!'})
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating company: {str(e)}', 'danger')
            return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': False, 'message': 'Method Not Allowed'}), 405

@bp.route('/company/delete/<int:company_id>', methods=['POST'])
@login_required
@superadmin_required
def delete_company(company_id):
    if request.method == "POST":
        company = Company.query.get_or_404(company_id)
        try:
            if company.user_account:
                db.session.delete(company.user_account)
            db.session.delete(company)
            db.session.commit()
            flash('Company deleted successfully!', 'success')
            return jsonify({'success': True, 'message': 'Company deleted successfully!'})
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting company: {str(e)}', 'danger')
            return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': False, 'message': 'Method Not Allowed'}), 405