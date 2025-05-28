# coorporate_app/app/routes/superadmin_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models import Company, Users, Personnels # Import models
from app import db # Import instance db
from app.utils.decorators import superadmin_required # Decorators
from datetime import datetime, timedelta
from sqlalchemy import func, cast, Date # Untuk fungsi database

bp = Blueprint('superadmin', __name__, template_folder='../templates/superadmin')

@bp.route('/dashboard')
@login_required
@superadmin_required
def dashboard():
    today = datetime.utcnow()
    last_7_days = today - timedelta(days=7)
    last_30_days = today - timedelta(days=30)

    total_companies = Company.query.count()
    companies_last_7_days = Company.query.filter(Company.createdAt >= last_7_days).count()
    companies_last_30_days = Company.query.filter(Company.createdAt >= last_30_days).count()

    total_accounts = Users.query.count()
    accounts_last_7_days = Users.query.filter(Users.createdAt >= last_7_days).count() # date_joined -> createdAt
    accounts_last_30_days = Users.query.filter(Users.createdAt >= last_30_days).count()

    total_employees = Personnels.query.count()
    employees_last_7_days = Personnels.query.filter(Personnels.createdAt >= last_7_days).count()
    employees_last_30_days = Personnels.query.filter(Personnels.createdAt >= last_30_days).count()

    # Fetch all companies for the list (if used directly on dashboard, otherwise move to company_list)
    companies_list = Company.query.all()

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
        'companies': companies_list,
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
        companies_query = companies_query.filter(Company.name.ilike(f'%{search_term}%')) # ilike for case-insensitive LIKE

    # Implement pagination manually or use Flask-SQLAlchemy-Pagination extension
    # For now, a simple slice
    total_count = companies_query.count()
    companies = companies_query.offset((page - 1) * entries_per_page).limit(entries_per_page).all()
    
    # Calculate total pages
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

        # Check for existing username or email
        if Users.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists.'}), 409
        if email and Users.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already exists.'}), 409

        try:
            # Create CustomUser (admin role)
            new_user = Users(username=username, email=email, role=Users.ROLE_ADMIN)
            new_user.set_password(password) # Hash password
            db.session.add(new_user)
            db.session.flush() # Commit user to get its ID before linking to company

            # Create Company and link to user
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
        # Periksa apakah superadmin yang login punya izin untuk mengedit company ini
        # (meskipun superadmin_required sudah cukup, bisa ada logic tambahan jika perlu)

        company_name = request.form.get('company_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password') # Password baru

        try:
            # Update Company name
            company.name = company_name if company_name else company.name

            # Update associated user details
            user_account = company.user_account # Access linked CustomUser
            if user_account:
                user_account.username = username if username else user_account.username
                user_account.email = email if email else user_account.email
                if password: # Update password only if provided
                    user_account.set_password(password)
                db.session.add(user_account) # Mark for update
            
            db.session.add(company) # Mark for update
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
            # SQLAlchemy CASCADE handles deletion of related entities if configured
            # Delete the associated user account if it exists
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