# coorporate_app/app/utils/decorators.py
from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user, login_required

def role_required(roles):
    def decorator(f):
        @wraps(f)
        @login_required # Pastikan user sudah login
        def decorated_function(*args, **kwargs):
            if current_user.role not in roles:
                flash('You do not have permission to access this page.', 'danger')
                # Redirect ke dashboard yang sesuai atau ke halaman utama
                if current_user.is_superadmin:
                    return redirect(url_for('superadmin.dashboard'))
                elif current_user.is_admin:
                    return redirect(url_for('admin.dashboard')) # Perhatikan: 'admin' bukan 'admin_panel' di blueprint
                elif current_user.is_employee:
                    return redirect(url_for('employee.dashboard'))
                else:
                    return redirect(url_for('auth.login')) # Fallback
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def superadmin_required(f):
    return role_required(['superadmin'])(f)

def admin_required(f):
    return role_required(['admin', 'superadmin'])(f)

def employee_required(f):
    return role_required(['employee', 'admin', 'superadmin'])(f)