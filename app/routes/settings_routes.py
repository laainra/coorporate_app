# coorporate_app/app/routes/settings_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models import Users # Import models
from app import db # Import instance db
from app.utils.decorators import admin_required, employee_required, superadmin_required # Decorators

bp = Blueprint('settings', __name__, template_folder='../templates') # Templates like profile.html might be here or in a separate profile blueprint

@bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = current_user # Users object
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        try:
            # Update username and email
            user.username = username if username else user.username
            user.email = email if email else user.email

            # Update password if provided
            if old_password or new_password or confirm_password:
                if not user.check_password(old_password):
                    flash('Old password is incorrect.', 'danger')
                    return render_template('profile.html', user=user)
                if new_password != confirm_password:
                    flash('New password and confirm password do not match.', 'danger')
                    return render_template('profile.html', user=user)
                if not new_password:
                    flash('New password cannot be empty.', 'danger')
                    return render_template('profile.html', user=user)
                
                user.set_password(new_password)
            
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('settings.profile'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'danger')
            return render_template('profile.html', user=user)
            
    return render_template('profile.html', user=user)

# No other specific settings routes from your Django code snippet.
# If you have general settings for company/admin, they would go under admin/superadmin blueprints.