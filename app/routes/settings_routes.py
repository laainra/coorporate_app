

# coorporate_app/app/routes/settings_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models import Users # Import models
from app import db # Import instance db
from app.utils.decorators import admin_required, employee_required, superadmin_required # Decorators

bp = Blueprint('settings', __name__, template_folder='../templates') # Templates like profile.html might be here or in a separate profile blueprint



def get_display_name_and_related_objects(user):
    """Helper untuk mendapatkan nama tampilan dan objek terkait."""
    display_name = user.username # Default
    company_obj = None
    personnel_obj = None

    if user.role == 'superadmin':
        display_name = "Super Admin"
    elif user.role == 'admin':
        if hasattr(user, 'company') and user.company: # Asumsi user.company adalah relasi
            company_obj = user.company
            display_name = company_obj.name if company_obj.name else f"{user.username} (Admin)"
        else:
            display_name = f"{user.username} (Admin)"
    elif user.role == 'employee':
        if hasattr(user, 'personnel_record') and user.personnel_record: # Asumsi user.personnel_record adalah relasi
            personnel_obj = user.personnel_record
            display_name = personnel_obj.name if personnel_obj.name else f"{user.username} (Employee)"
        else:
            display_name = f"{user.username} (Employee)"
    return display_name, company_obj, personnel_obj

@bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = current_user
    display_name, company, personnel = get_display_name_and_related_objects(user)
    form_data_on_error = {}

    if request.method == 'POST':
        form_data_on_error = request.form 
        
        validation_errors = []
        
        username_to_update = None
        email_to_update = None
        password_to_set = None 
        company_name_to_update = None
        personnel_name_to_update = None

        submitted_username = request.form.get('username')
        if submitted_username is not None: 
            cleaned_username = submitted_username.strip()
            if not cleaned_username:
                validation_errors.append(('Username cannot be empty.', 'danger'))
            elif cleaned_username != user.username:
    
                username_to_update = cleaned_username

        # 2. Validasi Email
        submitted_email = request.form.get('email')
        if submitted_email is not None: 
            cleaned_email = submitted_email.strip()
            if not cleaned_email:
                validation_errors.append(('Email cannot be empty.', 'danger'))

            elif cleaned_email != user.email:

                email_to_update = cleaned_email
        
        # 3. Validasi Password
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if old_password:
            if not user.check_password(old_password):
                validation_errors.append(('Old password is incorrect.', 'danger'))
            else:

                if new_password:
                    if not confirm_password:
                        validation_errors.append(('Please confirm your new password.', 'danger'))
                    elif new_password != confirm_password:
                        validation_errors.append(('New password and confirm password do not match.', 'danger'))
                    else:
                       
                        password_to_set = new_password

        elif new_password or confirm_password: 
          
            validation_errors.append(('Old password is required to set or change your password.', 'danger'))
        

        if user.role == 'admin' and company:
            submitted_company_name = request.form.get('company_name')
            if submitted_company_name is not None: # Jika field ada di form
                cleaned_company_name = submitted_company_name.strip()
                if not cleaned_company_name:
                    validation_errors.append(('Company name cannot be empty.', 'danger'))
                elif cleaned_company_name != company.name:
                    company_name_to_update = cleaned_company_name
        
        elif user.role == 'employee' and personnel:
            submitted_personnel_name = request.form.get('personnel_name')
            if submitted_personnel_name is not None: # Jika field ada di form
                cleaned_personnel_name = submitted_personnel_name.strip()
                if not cleaned_personnel_name:
                    validation_errors.append(('Personnel name cannot be empty.', 'danger'))
                elif cleaned_personnel_name != personnel.name:
                    personnel_name_to_update = cleaned_personnel_name

        if validation_errors:
            for msg, category in validation_errors:
                flash(msg, category)

            return render_template('profile.html',
                                   user=user,
                                   display_name=display_name,
                                   company=company,
                                   personnel=personnel,
                                   form_data=form_data_on_error) 
        else:

            db_commit_needed = False

            if username_to_update is not None:
                user.username = username_to_update
                db_commit_needed = True
            
            if email_to_update is not None:
                user.email = email_to_update
                db_commit_needed = True

            if password_to_set is not None: 
                user.set_password(password_to_set)
                db_commit_needed = True

            if company_name_to_update is not None and company:
                company.name = company_name_to_update
                db_commit_needed = True
            
            if personnel_name_to_update is not None and personnel:
                personnel.name = personnel_name_to_update
                db_commit_needed = True


            if db_commit_needed:
                try:
                    db.session.commit()
                    flash('Your profile has been updated successfully.', 'success')
                except Exception as e:
                    db.session.rollback()
                    flash(f'Error updating profile: {str(e)}', 'danger')

            else:
                flash('No changes were made to the profile.', 'info')
            
            return redirect(url_for('settings.profile')) 

    return render_template('profile.html',
                           user=user,
                           display_name=display_name,
                           company=company,
                           personnel=personnel,
                           form_data=None)