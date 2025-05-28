# coorporate_app/app/routes/auth_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from app.models import Users 
from app import db # Import instance db
from werkzeug.security import check_password_hash


bp = Blueprint('auth', __name__, template_folder='../templates') # Template folder untuk login.html

@bp.route('/')
def index():
    """Redirects the root URL to the login page."""
    return redirect(url_for('auth.login'))

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        # Redirect user yang sudah login sesuai rolenya
        if current_user.is_superadmin:
            return redirect(url_for('superadmin.dashboard'))
        elif current_user.is_admin:
            return redirect(url_for('admin.dashboard'))
        elif current_user.is_employee:
            return redirect(url_for('employee.dashboard'))
        else:
            flash('Your role is not recognized. Please contact support.', 'warning')
            logout_user() # Log out unrecognized roles
            return redirect(url_for('auth.login'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = Users.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)# Login user menggunakan Flask-Login
            flash(f'Welcome, {user.username}!', 'success')
            
            # Redirect berdasarkan role
            if user.is_superadmin:
                return redirect(url_for('superadmin.dashboard'))
            elif user.is_admin:
                return redirect(url_for('admin.dashboard'))
            elif user.is_employee:
                return redirect(url_for('employee.dashboard'))
            else:
                flash('Unauthorized role. Please contact support.', 'danger')
                logout_user()
                return redirect(url_for('auth.login'))
        else:
            flash('Invalid username or password.', 'danger') # Gunakan flash message
            return render_template('login.html', error_message='Invalid credentials')
            
    # Untuk GET request
    return render_template('login.html')

@bp.route('/logout')
@login_required # Pastikan user sudah login untuk logout
def logout():
    logout_user() # Logout user menggunakan Flask-Login
    flash('You have been logged out.', 'info') # Gunakan flash message
    return redirect(url_for('auth.login'))