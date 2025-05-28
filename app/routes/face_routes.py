# coorporate_app/app/routes/face_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from app.models import Personnels, Personnel_Images # Import models
from app import db # Import instance db
from app.utils.decorators import employee_required, admin_required # Decorators
from app.utils.config_variables import get_personnel_folder_path
from app.ai.face_recognition import _train_face_model_logic, capture_faces_logic # Import AI logic
import os
import shutil # Untuk operasi file/folder

bp = Blueprint('face', __name__, template_folder='../templates/employee_panel') # Templates related to face/dataset in employee_panel

@bp.route('/capture')
@login_required
@employee_required
def capture_page():
    """Renders the page for capturing new face images for a personnel."""
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        flash("Personnel data not found for your user account. Please contact admin.", "danger")
        return redirect(url_for('employee.dashboard'))
    return render_template('employee_panel/capture.html', name=personnel.name, personnel_id=personnel.id)

@bp.route('/capture_data', methods=['POST'])
@login_required
@employee_required
def capture_faces():
    """Endpoint to capture face images from webcam and save them for personnel dataset."""
    personnel = Personnels.query.filter_by(user_account=current_user).first()
    if not personnel:
        return jsonify({'status': 'error', 'message': 'User personnel data not found.'}), 400

    # Call the core logic, which handles webcam access and saving
    result = capture_faces_logic(personnel) # This function is in app.ai.face_recognition
    
    if result['status'] == 'success':
        flash(result['message'], 'success')
    else:
        flash(result['message'], 'danger')
        
    return jsonify(result)

@bp.route('/train')
@login_required
@admin_required # Only admin/superadmin can train the model
def train_model():
    """Endpoint to trigger the training of the face recognition model."""
    result = _train_face_model_logic() # Call the core AI logic
    if result['success']:
        flash(result['message'], 'success')
    else:
        flash(result['message'], 'danger')
    return redirect(url_for('face.dataset_no_id')) # Redirect to dataset management page (admin view)

@bp.route('/dataset')
@bp.route('/dataset/<int:personnel_id>')
@login_required
@admin_required # Admin can see all, employee can only see their own via /employee/dataset (or you merge logic here)
def dataset_no_id(personnel_id=None):
    """Displays the dataset of face images for a given personnel."""
    current_personnel = None
    if personnel_id: # Admin is viewing a specific personnel's dataset
        current_personnel = Personnels.query.get_or_404(personnel_id)
        # Check ownership for admin
        if current_personnel.company_obj != current_user.company_linked:
            flash("Unauthorized access to this personnel's dataset.", "danger")
            return redirect(url_for('admin.dashboard')) # Redirect if not authorized
    else: # If no personnel_id, assume employee accessing their own
        current_personnel = Personnels.query.filter_by(user_account=current_user).first()
        if not current_personnel:
            flash("Personnel data not found for your account.", "danger")
            return redirect(url_for('employee.dashboard')) # Redirect if employee has no personnel data

    images = []
    personnel_folder = os.path.join(get_personnel_folder_path(), current_personnel.name)
    if os.path.exists(personnel_folder):
        for file_name in os.listdir(personnel_folder):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                # Construct URL relative to Flask's static directory
                # Example: /static/img/personnel_pics/Name/image.jpg
                # image_path in DB is relative to UPLOAD_FOLDER (e.g., 'personnel_images/Name/file.jpg')
                # UPLOAD_FOLDER is app/static/img
                # So, the URL needs to be /static/img/personnel_images/Name/file.jpg
                # The image_path stored in DB is relative to UPLOAD_FOLDER (e.g. 'personnel_pics/Name/face_1_Name_1.jpg')
                relative_to_static_url = os.path.relpath(os.path.join(personnel_folder, file_name), current_app.root_path).replace(os.sep, "/")
                images.append({'url': url_for('static', filename=relative_to_static_url.replace('static/', '', 1)), 'filename': file_name}) # Remove 'static/' prefix
                # Example: 'app/static/img/personnel_pics/Name/image.jpg' -> 'img/personnel_pics/Name/image.jpg'
                # The 'img' part is already in UPLOAD_FOLDER, so relative_to_static_url will be like 'static/img/personnel_pics/...'
                # We need to remove the first 'static/' to make it work with url_for('static', filename=...)
                # Corrected:
                # Assuming UPLOAD_FOLDER = os.path.join(basedir, 'app', 'static','img')
                # Then personnel_folder is os.path.join(UPLOAD_FOLDER, 'personnel_pics', personnel.name)
                # So relative_path_for_static = os.path.relpath(os.path.join(personnel_folder, file_name), os.path.join(current_app.root_path, 'app', 'static')).replace(os.sep, "/")
                # images.append({'url': url_for('static', filename=relative_path_for_static), 'filename': file_name})

                # Let's simplify the URL generation for images assuming UPLOAD_FOLDER is served as static.
                # If UPLOAD_FOLDER is `app/static/img`, then image_path stored in DB is `personnel_pics/Name/image.jpg`
                # The URL will be `/static/img/personnel_pics/Name/image.jpg`
                images.append({
                    'url': url_for('static', filename=os.path.join('img', os.path.relpath(os.path.join(personnel_folder, file_name), current_app.config['UPLOAD_FOLDER']))),
                    'filename': file_name,
                    'personnel_name': current_personnel.name # Add personnel_name for deletion logic
                })


    return render_template('employee_panel/dataset.html', images=images, name=current_personnel.name, personnel=current_personnel)


@bp.route('/delete_images', methods=['POST'])
@login_required
@admin_required # Admin can delete any, employee can delete their own (logic needs refinement if employee deletes)
def delete_images():
    if request.method == 'POST':
        # Assuming images_to_delete is a list of filenames
        images_to_delete = request.form.getlist('images_to_delete')
        personnel_name = request.form.get('personnel_name') # Get personnel name from form

        personnel = Personnels.query.filter_by(name=personnel_name).first()
        if not personnel:
            flash("Personnel not found.", "danger")
            return redirect(request.referrer or url_for('admin.dashboard')) # Redirect back

        # Check ownership for admin (if admin, can delete any in their company)
        if current_user.is_admin and personnel.company_obj != current_user.company_linked:
            flash("Unauthorized: Personnel not in your company.", "danger")
            return redirect(request.referrer or url_for('admin.dashboard'))
        # If employee, ensure they are deleting their own images
        if current_user.is_employee and personnel.user_account != current_user:
            flash("Unauthorized: You can only delete your own images.", "danger")
            return redirect(request.referrer or url_for('employee.dashboard'))


        deleted_count = 0
        for filename in images_to_delete:
            # Reconstruct absolute path to the image file
            full_path = os.path.join(get_personnel_folder_path(), personnel.name, filename)
            
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    # Also delete from database
                    # The image_path stored in DB is relative to UPLOAD_FOLDER
                    relative_path_in_db = os.path.relpath(full_path, current_app.config['UPLOAD_FOLDER']).replace("\\", "/")
                    img_db_entry = Personnel_Images.query.filter_by(image_path=relative_path_in_db).first()
                    if img_db_entry:
                        db.session.delete(img_db_entry)
                        db.session.commit()
                    deleted_count += 1
                except Exception as e:
                    flash(f"Error deleting {filename}: {str(e)}", "warning")
                    db.session.rollback()

        flash(f"Deleted {deleted_count} selected images.", "success")
        return redirect(request.referrer or url_for('face.dataset_no_id', personnel_id=personnel.id))
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405