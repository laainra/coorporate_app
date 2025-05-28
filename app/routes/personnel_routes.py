# coorporate_app/app/routes/personnel_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from app.models import Personnels, Divisions, Company, Users, Personnel_Images, Personnel_Entries # Import models
from app import db # Import instance db
from app.utils.decorators import admin_required # Decorators
from app.utils.config_variables import get_personnel_folder_path # Import var equivalent
from datetime import datetime
from sqlalchemy import func, and_ # Untuk fungsi database
import os
import shutil
import pandas as pd
from werkzeug.utils import secure_filename
import io # For Excel file generation

# For face image processing (from your AI code)
from app.ai.face_recognition import _train_face_model_logic # Import the training logic
# Note: RV.get_feature and RV.known_features were from Django.
# In Flask, RV.get_feature should be in app.ai.face_recognition.py and operate on image files.
# RV.known_features implies a global state. In Flask, for production, you'd load/manage this in a smarter way (e.g., Redis, a dedicated service, or on app startup).
# For now, I'll remove RV specific calls that imply complex global state.
# For simplicity, add_personnel_image will just save the image and link it to personnel_images,
# and training model will pick up all images in the personnel_pics folder.

bp = Blueprint('personnel', __name__, template_folder='../templates/admin_panel') # Templates for personnel management are in admin_panel

@bp.route('/home')
@login_required
@admin_required
def personnels():
    company = Company.query.filter_by(user_account=current_user).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))

    personnels_list = Personnels.query.filter_by(company_obj=company).all()
    divisions_list = Divisions.query.filter_by(company_obj=company).all()

    personnel_data = []
    for personnel in personnels_list:
        # Fetch images for each personnel
        images = Personnel_Images.query.filter_by(personnel_obj=personnel).all()

        # Check if images exist, otherwise use a default image path
        profile_image_url = 'static/img/user_default.png' # Default path relative to static
        if images:
            # Assuming image_path in model is relative to UPLOAD_FOLDER
            # UPLOAD_FOLDER = app/static/img, image_path = personnel_images/name/file.jpg
            # So, f'img/{img.image_path}' might become f'{current_app.config["UPLOAD_FOLDER"]}/personnel_pics/{personnel.name}/{os.path.basename(img.image_path)}'
            # Or, if image_path stored in DB is relative to app.static folder (e.g. img/personnel_images/...)
            # We'll use url_for('static', filename=...) in template for consistency.
            # Here, we store the full relative path from static folder for easier template rendering
            first_image_path = images[0].image_path # This path is relative to UPLOAD_FOLDER (e.g. personnel_images/Name/face_1_Name_1.jpg)
            profile_image_url = f"img/{first_image_path}" # Assuming UPLOAD_FOLDER is app/static/img and image_path is relative to app/static/img
            
        personnel_data.append({
            'id': personnel.id,
            'name': personnel.name,
            'username': personnel.user_account.username if personnel.user_account else 'N/A',
            'email': personnel.user_account.email if personnel.user_account else 'N/A',
            'division': personnel.division_obj.name if personnel.division_obj else 'N/A',
            # 'gender': personnel.gender, # Uncomment if you enable these fields in Personnel model
            # 'employment_status': personnel.employment_status, # Uncomment if you enable these fields
            'profile_image': profile_image_url,
        })
    
    divisions_data = [{'id': div.id, 'name': div.name} for div in divisions_list]

    return render_template('admin_panel/personnels.html', Personnels=personnel_data, Divisions=divisions_data, Page="Personnels")

@bp.route('/<int:personnel_id>')
@login_required
@admin_required
def get_personnel(personnel_id):
    personnel = Personnels.query.get_or_404(personnel_id)
    # Check ownership
    if personnel.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to personnel'}), 403
    
    data = {
        'id': personnel.id,
        'name': personnel.name,
        'division': {
            'id': personnel.division_obj.id if personnel.division_obj else None,
            'name': personnel.division_obj.name if personnel.division_obj else 'N/A'
        },
        'email': personnel.user_account.email if personnel.user_account else 'N/A',
        'username': personnel.user_account.username if personnel.user_account else 'N/A',
        'password': '********', # Never send actual password
    }
    return jsonify(data)

@bp.route('/add', methods=['POST'])
@login_required
@admin_required
def add_personnel():
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        division_id = request.form.get('division')
        # gender = request.form.get('gender') # Uncomment if used
        # employment_status = request.form.get('employment_status') # Uncomment if used

        company = Company.query.filter_by(user_account=current_user).first()
        if not company:
            return jsonify({'status': 'error', 'message': 'Admin account not linked to a company'}), 403

        division = Divisions.query.get(division_id)
        if not division or division.company_obj != company: # Ensure division belongs to the same company
            return jsonify({'status': 'error', 'message': 'Invalid division selected'}), 400

        # Ensure personnel name is unique within the company
        existing_personnel = Personnels.query.filter_by(name=name, company_obj=company).first()
        if existing_personnel:
            return jsonify({'status': 'error', 'message': 'Personnel with this name already exists in your company'}), 409
        
        # Ensure username is unique globally for Users
        existing_user = Users.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'status': 'error', 'message': 'Username already taken'}), 409
        
        # Ensure email is unique globally for Users (if email is not null)
        if email and Users.query.filter_by(email=email).first():
            return jsonify({'status': 'error', 'message': 'Email already registered'}), 409

        try:
            # Create CustomUser for personnel
            new_user = Users(username=username, email=email, role=Users.ROLE_EMPLOYEE)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.flush() # Get ID for new_user

            # Create personnel entry
            new_personnel = Personnels(
                name=name,
                user_id=new_user.id,
                division_id=division.id,
                company_id=company.id,
                # gender=gender, # Uncomment if used
                # employment_status=employment_status # Uncomment if used
            )
            db.session.add(new_personnel)
            db.session.commit()

            # Create personnel image folder
            personnel_folder_path = os.path.join(get_personnel_folder_path(), name)
            os.makedirs(personnel_folder_path, exist_ok=True)
            
            flash('Personnel added successfully!', 'success')
            return jsonify({'status': 'success', 'message': 'Personnel added successfully'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': f'Failed to add personnel: {str(e)}'}), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@bp.route('/edit/<int:personnel_id>', methods=['POST'])
@login_required
@admin_required
def edit_personnel(personnel_id):
    if request.method == 'POST':
        personnel = Personnels.query.get_or_404(personnel_id)
        # Check ownership
        if personnel.company_obj != current_user.company_linked:
            return jsonify({'status': 'error', 'message': 'Unauthorized access to personnel'}), 403
            
        old_name = personnel.name # Simpan nama lama untuk operasi folder
        
        name = request.form.get('name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password') # Password baru (optional)
        division_id = request.form.get('division')
        # gender = request.form.get('gender') # Uncomment if used
        # employment_status = request.form.get('employment_status') # Uncomment if used

        division = Divisions.query.get(division_id)
        if not division or division.company_obj != personnel.company_obj:
            return jsonify({'status': 'error', 'message': 'Invalid division selected'}), 400

        try:
            # Update CustomUser data
            user_account = personnel.user_account
            if user_account:
                user_account.username = username if username else user_account.username
                user_account.email = email if email else user_account.email
                if password: # Update password only if provided
                    user_account.set_password(password)
                db.session.add(user_account) # Mark for update

            # Update Personnel data
            personnel.name = name if name else personnel.name
            personnel.division_id = division.id
            # personnel.gender = gender # Uncomment if used
            # personnel.employment_status = employment_status # Uncomment if used
            db.session.add(personnel) # Mark for update
            
            # Rename personnel folder if name changed
            if old_name != personnel.name:
                old_path = os.path.join(get_personnel_folder_path(), old_name)
                new_path = os.path.join(get_personnel_folder_path(), personnel.name)
                if os.path.exists(old_path):
                    shutil.move(old_path, new_path)
                    # Update image_path in Personnel_Images to reflect new folder name
                    for img in Personnel_Images.query.filter_by(personnel_obj=personnel).all():
                        img.image_path = os.path.relpath(os.path.join(new_path, os.path.basename(img.image_path)), current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
                        db.session.add(img)

            db.session.commit()
            flash('Personnel updated successfully!', 'success')
            return jsonify({'status': 'success', 'message': 'Personnel updated successfully'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': f'Failed to update personnel: {str(e)}'}), 500
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405


@bp.route('/delete/<int:personnel_id>', methods=['POST'])
@login_required
@admin_required
def delete_personnel(personnel_id):
    personnel = Personnels.query.get_or_404(personnel_id)
    # Check ownership
    if personnel.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to personnel'}), 403
        
    try:
        # Delete personnel folder and its contents
        personnel_folder_path = os.path.join(get_personnel_folder_path(), personnel.name)
        if os.path.exists(personnel_folder_path):
            shutil.rmtree(personnel_folder_path)

        # Delete associated user account
        if personnel.user_account:
            db.session.delete(personnel.user_account)

        # SQLAlchemy cascade takes care of Personnel_Images, Personnel_Entries, Work_Timer
        db.session.delete(personnel)
        db.session.commit()
        flash('Personnel deleted successfully!', 'success')
        return jsonify({'status': 'success', 'message': 'Personnel deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Failed to delete personnel: {str(e)}'}), 500


@bp.route('/attendance/<int:personnel_id>')
@login_required
@admin_required
def attendance_details(personnel_id):
    personnel = Personnels.query.get_or_404(personnel_id)
    # Check ownership
    if personnel.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to personnel'}), 403

    # Query attendance entries for the personnel
    entries = Personnel_Entries.query.filter_by(personnel_obj=personnel).all()

    total_presence = len(entries)
    total_ontime = sum(1 for e in entries if e.presence_status == 'ONTIME')
    total_late = sum(1 for e in entries if e.presence_status == 'LATE')
    total_absence = sum(1 for e in entries if e.presence_status == 'UNKNOWN') # Assuming UNKNOWN means absence

    data = {
        'name': personnel.name,
        'total_presence': total_presence,
        'total_ontime': total_ontime,
        'total_late': total_late,
        'total_absence': total_absence,
    }
    return jsonify(data)

@bp.route('/images/<int:personnel_id>')
@login_required
@admin_required
def get_personnel_images(personnel_id):
    personnel = Personnels.query.get_or_404(personnel_id)
    # Check ownership
    if personnel.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to personnel images'}), 403
    
    images = Personnel_Images.query.filter_by(personnel_obj=personnel).all()
    
    image_list = []
    for img in images:
        # Assuming image_path in DB is relative to UPLOAD_FOLDER (e.g., personnel_images/Name/image.jpg)
        # We want a URL that can be used in templates (e.g., static/img/personnel_images/Name/image.jpg)
        relative_to_upload_folder = img.image_path # Example: 'personnel_images/Name/face_1_Name_1.jpg'
        full_static_url = url_for('static', filename=f"img/{relative_to_upload_folder}") # Adjust 'img' if UPLOAD_FOLDER is not 'app/static/img'
        image_list.append({'id': img.id, 'image_path': full_static_url, 'raw_path': img.image_path}) # raw_path for deletion/move operations
    
    return jsonify({'images': image_list})


@bp.route('/images/add/<int:personnel_id>', methods=['POST'])
@login_required
@admin_required
def add_personnel_image(personnel_id):
    personnel = Personnels.query.get_or_404(personnel_id)
    # Check ownership
    if personnel.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to personnel'}), 403

    no_face_count = 0

    if request.method == 'POST' and request.files:
        files = request.files.getlist('images')

        if not files:
            return jsonify({'status': 'error', 'message': 'No images uploaded'}), 400

        personnel_folder = os.path.join(get_personnel_folder_path(), personnel.name)
        os.makedirs(personnel_folder, exist_ok=True)

        for x, f in enumerate(files):
            if f.filename == '':
                continue # Skip empty file inputs

            filename = f"face_{personnel.id}_{personnel.name}_{int(datetime.now().timestamp())}_{x}.jpg" # Unique filename
            file_path_abs = os.path.join(personnel_folder, filename)

            f.save(file_path_abs) # Save uploaded file temporarily

            img = cv2.imread(file_path_abs)
            if img is None:
                os.remove(file_path_abs)
                no_face_count += 1
                continue
            
            # Use Haar Cascade to detect faces
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_in_img = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

            if len(faces_in_img) == 1:
                # Store relative path for database
                # personnel_images/Name/filename.jpg
                relative_path_to_upload_folder = os.path.relpath(file_path_abs, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
                
                new_personnel_image = Personnel_Images(
                    personnel_id=personnel.id,
                    image_path=relative_path_to_upload_folder
                )
                db.session.add(new_personnel_image)
                # db.session.commit() # Commit in loop for immediate saving or outside for bulk
            else:
                os.remove(file_path_abs) # Delete if no face or multiple faces
                no_face_count += 1
        
        db.session.commit() # Commit all new images in one go

        if no_face_count == len(files):
            return jsonify({'status': 'error', 'message': 'No valid faces detected in the uploaded images or images corrupted.'}), 400

        flash('Images added successfully!', 'success')
        return jsonify({'status': 'success', 'message': 'Images added successfully'})

    return jsonify({'status': 'error', 'message': 'Invalid request'}), 405


@bp.route('/images/delete/<int:image_id>', methods=['POST'])
@login_required
@admin_required
def delete_personnel_image(image_id):
    image = Personnel_Images.query.get_or_404(image_id)
    # Check ownership
    if image.personnel_obj.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to image'}), 403

    try:
        # Reconstruct absolute path
        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image.image_path)
        if os.path.exists(full_path):
            os.remove(full_path)
        
        db.session.delete(image)
        db.session.commit()
        flash('Image deleted successfully!', 'success')
        return jsonify({'status': 'success', 'message': 'Image deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Failed to delete image: {str(e)}'}), 500


@bp.route('/images/move', methods=['POST'])
@login_required
@admin_required
def move_personnel_image():
    if request.method == 'POST':
        try:
            data = request.get_json() # Get JSON data
            image_id = data.get('image_id')
            new_personnel_id = data.get('new_personnel_id')
            
            image = Personnel_Images.query.get_or_404(image_id)
            new_personnel = Personnels.query.get_or_404(new_personnel_id)

            # Check ownership for both source and destination
            if image.personnel_obj.company_obj != current_user.company_linked or \
               new_personnel.company_obj != current_user.company_linked:
                return jsonify({'status': 'error', 'message': 'Unauthorized access or personnel not in your company'}), 403
            
            old_full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image.image_path)
            
            # Construct new path based on new personnel's folder
            new_personnel_folder = os.path.join(get_personnel_folder_path(), new_personnel.name)
            os.makedirs(new_personnel_folder, exist_ok=True) # Ensure target folder exists
            
            new_full_path = os.path.join(new_personnel_folder, os.path.basename(old_full_path))
            
            shutil.move(old_full_path, new_full_path) # Move the file
            
            # Update database record
            image.personnel_id = new_personnel.id
            image.image_path = os.path.relpath(new_full_path, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
            db.session.commit()

            flash('Image moved successfully!', 'success')
            return jsonify({'status': 'success', 'message': 'Image moved successfully'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': f'Failed to move image: {str(e)}'}), 500
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@bp.route('/import', methods=['POST'])
@login_required
@admin_required
def import_personnel():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
        excel_file = request.files['file']
        if excel_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400
        
        if not excel_file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({'status': 'error', 'message': 'Invalid file type. Only Excel files are allowed.'}), 400

        try:
            df = pd.read_excel(excel_file)
            company = Company.query.filter_by(user_account=current_user).first()
            if not company:
                return jsonify({'status': 'error', 'message': 'Admin account not linked to a company'}), 403

            imported_count = 0
            errors = []

            for index, row in df.iterrows():
                try:
                    name = row.get('name')
                    username = row.get('username')
                    password = row.get('password')
                    email = row.get('email')
                    division_name = row.get('division')

                    if not all([name, username, password, division_name]):
                        errors.append(f"Row {index+1}: Missing required data. Skipping.")
                        continue

                    division = Divisions.query.filter_by(company_obj=company, name=division_name).first()
                    if not division:
                        errors.append(f"Row {index+1}: Division '{division_name}' not found in your company. Skipping.")
                        continue

                    # Check uniqueness for personnel name within company
                    if Personnels.query.filter_by(name=name, company_obj=company).first():
                        errors.append(f"Row {index+1}: Personnel with name '{name}' already exists in your company. Skipping.")
                        continue
                    
                    # Check uniqueness for username globally
                    if Users.query.filter_by(username=username).first():
                        errors.append(f"Row {index+1}: Username '{username}' already exists. Skipping.")
                        continue

                    # Check uniqueness for email globally (if email is provided)
                    if email and Users.query.filter_by(email=email).first():
                        errors.append(f"Row {index+1}: Email '{email}' already registered. Skipping.")
                        continue

                    new_user = Users(username=username, email=email, role=Users.ROLE_EMPLOYEE)
                    new_user.set_password(password)
                    db.session.add(new_user)
                    db.session.flush() # To get user ID

                    new_personnel = Personnels(
                        name=name,
                        user_id=new_user.id,
                        division_id=division.id,
                        company_id=company.id
                    )
                    db.session.add(new_personnel)
                    
                    personnel_folder_path = os.path.join(get_personnel_folder_path(), name)
                    os.makedirs(personnel_folder_path, exist_ok=True)
                    
                    imported_count += 1

                except Exception as e:
                    errors.append(f"Row {index+1}: Error processing data - {str(e)}. Skipping.")
                    db.session.rollback() # Rollback current transaction in case of error
                    continue # Continue to next row

            db.session.commit() # Commit all valid entries
            
            if errors:
                flash("Some personnels could not be imported. See errors below.", "warning")
                for err in errors:
                    flash(err, "warning")
            flash(f"Successfully imported {imported_count} new personnels!", "success")
            return jsonify({'status': 'success', 'message': f'Import completed. {imported_count} new personnels imported.', 'errors': errors})

        except Exception as e:
            flash(f'Error importing file: {str(e)}', 'danger')
            return jsonify({'status': 'error', 'message': f'Failed to import file: {str(e)}'}), 500
    
    return redirect(url_for('admin.employees')) # Redirect back to employees list

@bp.route('/download_template')
@login_required
@admin_required
def download_template():
    # Create an in-memory output file for Excel
    output = io.BytesIO()
    
    # Create a new workbook and add a worksheet.
    # Note: openpyxl is generally recommended for modern .xlsx files, but XlsxWriter is fine too.
    # If using openpyxl directly, the code would be slightly different.
    # Assuming XlsxWriter is installed as per requirements.
    from xlsxwriter.workbook import Workbook
    workbook = Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet('Employee Data')

    # Add header row
    columns = ['name', 'username', 'password', 'email', 'division']
    worksheet.write_row(0, 0, columns)

    # Close the workbook
    workbook.close()
    
    # Go back to the beginning of the stream
    output.seek(0)

    # Create a response with the Excel file
    response = current_app.response_class(output.read(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers.set('Content-Disposition', 'attachment', filename='employee_data_template.xlsx')
    return response

@bp.route('/entries_data')
@login_required
@admin_required
def personnel_entries_data():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Date range is required."}), 400
    
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    company = Company.query.filter_by(user_account=current_user).first()
    if not company:
        return jsonify({"error": "Admin account not linked to a company."}), 403

    # Query entries filtered by company and date range
    entries_query = Personnel_Entries.query.join(Personnels).filter(
        Personnels.company_obj == company,
        cast(Personnel_Entries.timestamp, Date).between(start_date, end_date)
    )

    # Aggregate counts
    late_count = entries_query.filter(Personnel_Entries.presence_status == 'LATE').count()
    ontime_count = entries_query.filter(Personnel_Entries.presence_status == 'ONTIME').count()
    
    # For total presence (ONTIME + LATE) and total absence (UNKNOWN)
    total_presence = entries_query.filter(
        (Personnel_Entries.presence_status == 'ONTIME') | (Personnel_Entries.presence_status == 'LATE')
    ).count()
    total_absence = entries_query.filter(Personnel_Entries.presence_status == 'UNKNOWN').count()

    return jsonify({
        "late_count": late_count,
        "ontime_count": ontime_count,
        "total_presence": total_presence,
        "total_absence": total_absence,
    })

@bp.route('/download_presence_report')
@login_required
@admin_required
def download_personnel_presence():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    selected_personnel_name = request.args.get('personnel_name', 'All') # Can be 'All' or specific name

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Date range is required."}), 400
    
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    company = Company.query.filter_by(user_account=current_user).first()
    if not company:
        return jsonify({"error": "Admin account not linked to a company."}), 403

    # Build the base query
    entries_query = Personnel_Entries.query.join(Personnels).filter(
        Personnels.company_obj == company,
        cast(Personnel_Entries.timestamp, Date).between(start_date, end_date)
    )

    if selected_personnel_name != 'All':
        entries_query = entries_query.filter(Personnels.name == selected_personnel_name)
    
    # Aggregate data by personnel
    # This involves complex group_by and aggregation in SQLAlchemy
    # Let's adapt the raw SQL logic to a more SQLAlchemy-friendly approach for aggregation
    # Alternatively, fetch all filtered entries and process with Pandas for complex aggregation
    
    # Simple approach: fetch all relevant entries and process in Python/Pandas
    all_filtered_entries = entries_query.order_by(Personnels.name, Personnel_Entries.timestamp).all()

    # Convert to a list of dicts for DataFrame
    data_for_df = []
    for entry in all_filtered_entries:
        data_for_df.append({
            'Personnel ID': entry.personnel_obj.id,
            'Personnel Name': entry.personnel_obj.name,
            'Date': entry.timestamp.date().isoformat(),
            'Time': entry.timestamp.time().strftime('%H:%M:%S'),
            'Presence Status': entry.presence_status,
            'Camera Name': entry.camera_obj.cam_name if entry.camera_obj else 'N/A',
            'Image Path': entry.image # This is relative path
        })
    
    if not data_for_df:
        return jsonify({"message": "No data found for the selected criteria."}), 200

    df = pd.DataFrame(data_for_df)
    
    # Create an in-memory output file
    output = io.BytesIO()
    
    # Use pandas to_excel for simplicity, it handles the writer
    df.to_excel(output, index=False, sheet_name='Attendance Report')
    output.seek(0) # Rewind to the beginning

    file_name_suffix = selected_personnel_name
    if selected_personnel_name == 'All':
        file_name_suffix = 'All_Personnel'

    response = current_app.response_class(output.read(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers.set('Content-Disposition', 'attachment', filename=f'Attendance_Report_{file_name_suffix}_{start_date_str}_to_{end_date_str}.xlsx')
    return response