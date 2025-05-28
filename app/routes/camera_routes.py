# coorporate_app/app/routes/camera_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user
from app.models import Camera_Settings, Company, Counted_Instances # Import models
from app import db # Import instance db
from app.utils.decorators import admin_required # Decorators
from app.ai.utils import get_camera_instance, release_camera_instance # For managing cv2.VideoCapture instances
from datetime import datetime, time # Untuk format waktu
import numpy as np # Untuk manipulasi array (polygon)

bp = Blueprint('camera', __name__, template_folder='../templates/admin_panel') # Templates in admin_panel

# Helper function to format time strings (from your Django code)
def format_time(time_str, default_time=None):
    """Convert 'HH:MM' to datetime.time object."""
    if not time_str:
        return default_time # Keep existing if not provided
    try:
        # Assuming input is HH:MM or HH:MM:SS
        if len(time_str) == 5: # HH:MM
            return datetime.strptime(time_str, "%H:%M").time()
        elif len(time_str) == 8: # HH:MM:SS
            return datetime.strptime(time_str, "%H:%M:%S").time()
        else:
            raise ValueError("Time string format not recognized.")
    except ValueError:
        print(f"Invalid time format: {time_str}. Using default.")
        return default_time # Return existing default on error


# @bp.route('/')
# @login_required
# @admin_required
# def camera():
#     company = Company.query.filter_by(user_account=current_user).first()
#     if not company:
#         flash("Admin account not linked to a company.", "danger")
#         return redirect(url_for('auth.login'))
    
#     # Retrieve all cameras for the current company
#     cams = Camera_Settings.query.filter_by(company_obj=company).all()

#     # Determine active_cam from session or first available camera
#     active_cam_id = session.get('cam_id')
#     active_cam = None
#     if active_cam_id:
#         active_cam = Camera_Settings.query.get(active_cam_id)
#         if not active_cam or active_cam.company_obj != company: # Validate ownership
#             active_cam = None
#             session.pop('cam_id', None) # Clear invalid session cam_id
    
#     if not active_cam and cams:
#         active_cam = cams[0] # Pick the first camera if no active_cam in session or it's invalid
#         session['cam_id'] = active_cam.id

#     # Try to get a frame for display size, if active_cam is valid and active
#     frame_shape = (0.1, 0.1) # Default
#     toggle_settings = session.get('toggle_polygon', False) # State of settings panel

#     if active_cam and active_cam.cam_is_active:
#         # Attempt to open the camera to get frame shape
#         cap = get_camera_instance(int(active_cam.feed_src) if active_cam.feed_src.isdigit() else active_cam.feed_src)
#         if cap and cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 frame_shape = (frame.shape[0], frame.shape[1])
#             # Do NOT release cap here, as it's managed by get_camera_instance cache.
#             # It will be released when stop_stream is called or app exits.
#         else:
#             flash(f"Warning: Camera '{active_cam.cam_name}' cannot be opened. Check source/activity.", "warning")
#             # Optionally set cam_is_active to False in DB if it can't open
#             active_cam.cam_is_active = False
#             db.session.commit()

#     # Pass data to template
#     return render_template('admin_panel/camera.html', 
#                            Cams=cams,
#                            Active_Cam=active_cam,
#                            Frame_X=frame_shape[1], # Width
#                            Frame_Y=frame_shape[0], # Height
#                            Toggle_Settings=toggle_settings,
#                            Page="Camera")

@bp.route('/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_camera():
    company = Company.query.filter_by(user_account=current_user).first()
    if not company:
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        cam_name = request.form.get('cam_name')
        feed_src = request.form.get('feed_src')
        role_camera = request.form.get('role_camera') # 'P' or 'T'

        if not all([cam_name, feed_src, role_camera]):
            flash('All fields are required.', 'danger')
            return render_template('admin_panel/add_camera.html') # Render form again with error

        # Optional: Validate feed_src can be opened before saving
        # cap_test = get_camera_instance(int(feed_src) if feed_src.isdigit() else feed_src)
        # if not cap_test or not cap_test.isOpened():
        #     flash("Failed to open camera feed source. Please check the source.", "danger")
        #     if cap_test: release_camera_instance(cap_test.source) # Release test instance
        #     return render_template('admin_panel/add_camera.html')
        # if cap_test: release_camera_instance(cap_test.source) # Release test instance

        new_cam = Camera_Settings(
            cam_name=cam_name,
            feed_src=feed_src,
            role_camera=role_camera,
            company_id=company.id, # Link to current admin's company
            cam_is_active=True # Set active on creation
        )
        db.session.add(new_cam)
        db.session.commit()
        flash('Camera added successfully!', 'success')
        return redirect(url_for('camera.camera')) # Redirect to camera list

    return render_template('admin_panel/add_camera.html', Active_Cam=None) # No active_cam for add page

@bp.route('/edit/<int:cam_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_camera(cam_id):
    camera = Camera_Settings.query.get_or_404(cam_id)
    # Check ownership
    if camera.company_obj != current_user.company_linked:
        flash("Unauthorized access to camera settings.", "danger")
        return redirect(url_for('camera.camera'))

    if request.method == 'POST':
        # Update fields from form data
        camera.cam_name = request.form.get('cam_name', camera.cam_name)
        new_feed_src = request.form.get('feed_src', camera.feed_src)
        camera.cam_is_active = request.form.get('cam_is_active') == 'on'
        camera.gender_detection = request.form.get('gender_detection') == 'on'
        camera.face_detection = request.form.get('face_detection') == 'on'
        camera.face_capture = request.form.get('face_capture') == 'on'
        camera.id_card_detection = request.form.get('id_card_detection') == 'on'
        camera.uniform_detection = request.form.get('uniform_detection') == 'on'
        camera.shoes_detection = request.form.get('shoes_detection') == 'on'
        camera.ciggerate_detection = request.form.get('ciggerate_detection') == 'on'
        camera.sit_detection = request.form.get('sit_detection') == 'on'
        camera.cam_start = request.form.get('cam_start', camera.cam_start)
        camera.cam_stop = request.form.get('cam_stop', camera.cam_stop)
        camera.attendance_time_start = request.form.get('attendance_time_start', camera.attendance_time_start)
        camera.attendance_time_end = request.form.get('attendance_time_end', camera.attendance_time_end)
        camera.leaving_time_start = request.form.get('leaving_time_start', camera.leaving_time_start)
        camera.leaving_time_end = request.form.get('leaving_time_end', camera.leaving_time_end)
        camera.role_camera = request.form.get('role_camera', camera.role_camera)

        # Handle feed_src change: release old stream if changed
        if camera.feed_src != new_feed_src:
            release_camera_instance(camera.feed_src) # Release old source
            camera.feed_src = new_feed_src # Update to new source
            # The new stream will be opened when its video_feed/presence_stream URL is accessed.

        db.session.commit()
        flash('Camera settings updated successfully!', 'success')
        return redirect(url_for('camera.camera'))
    
    return render_template('admin_panel/edit_camera.html', cam=camera)

@bp.route('/get_camera_data/<int:cam_id>')
@login_required
@admin_required
def get_camera_data(cam_id):
    camera = Camera_Settings.query.get_or_404(cam_id)
    if camera.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to camera settings'}), 403
    
    data = {
        'id': camera.id,
        'cam_name': camera.cam_name,
        'feed_src': camera.feed_src,
        'cam_is_active': camera.cam_is_active,
        'x1': camera.x1, 'y1': camera.y1, 'x2': camera.x2, 'y2': camera.y2,
        'x3': camera.x3, 'y3': camera.y3, 'x4': camera.x4, 'y4': camera.y4,
        'x5': camera.x5, 'y5': camera.y5, 'x6': camera.x6, 'y6': camera.y6,
        'x7': camera.x7, 'y7': camera.y7, 'x8': camera.x8, 'y8': camera.y8,
        'gender_detection': camera.gender_detection,
        'face_detection': camera.face_detection,
        'face_capture': camera.face_capture,
        'id_card_detection': camera.id_card_detection,
        'uniform_detection': camera.uniform_detection,
        'shoes_detection': camera.shoes_detection,
        'ciggerate_detection': camera.ciggerate_detection,
        'sit_detection': camera.sit_detection,
        'cam_start': camera.cam_start,
        'cam_stop': camera.cam_stop,
        'attendance_time_start': camera.attendance_time_start,
        'attendance_time_end': camera.attendance_time_end,
        'leaving_time_start': camera.leaving_time_start,
        'leaving_time_end': camera.leaving_time_end,
        'role_camera': camera.role_camera
    }
    return jsonify({'status': 'success', 'data': data})

@bp.route('/save_coordinates', methods=['POST'])
@login_required
@admin_required
def save_coordinates():
    data = request.json # Expecting JSON data
    cam_id = data.get('camera_id')
    coordinates = data.get('coordinates') # A dictionary with x1, y1, ..., x8, y8

    camera = Camera_Settings.query.get_or_404(cam_id)
    if camera.company_obj != current_user.company_linked:
        return jsonify({'status': 'error', 'message': 'Unauthorized access to camera settings'}), 403

    try:
        # Convert values to int as they come from JS
        for key, value in coordinates.items():
            setattr(camera, key, int(value)) # Set attributes dynamically
        db.session.commit()
        
        # Update polygon in the running stream (if MC existed, this would be MC.set_polygon)
        # For our Flask setup, the stream generator reads directly from DB on each frame or on startup.
        # So, no need to call MC.set_polygon directly.
        # If the stream is active, it might need to be restarted to pick up new coordinates.
        flash('Coordinates saved successfully!', 'success')
        return jsonify({'status': 'success', 'message': 'Coordinates saved successfully!'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Failed to save coordinates: {str(e)}'}), 500


# ====================================================================
# Specific Add/Edit routes for Tracking and Presence Cameras (from your Django code)
# ====================================================================

@bp.route('/add_tracking_camera', methods=['GET', 'POST'])
@login_required
# @admin_required # Ensure this decorator is correctly defined and working
def add_tracking_camera():
    # Use company_obj to avoid confusion with the query object
    company_obj = Company.query.filter_by(user_id=current_user.id).first()
    
    if not company_obj:
        # For AJAX POST requests, return a JSON error
        if request.method == 'POST':
            return jsonify({'status': 'error', 'message': 'Admin account not linked to a company.'}), 403 # Forbidden or 400 Bad Request
        # For GET requests or traditional form submissions (not via this JS fetch)
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login')) 
    
    if request.method == 'POST':
        cam_name = request.form.get('cam_name')
        camera_type = request.form.get('camera_type')
        feed_src_rtsp = request.form.get('feed_src_rtsp')
        
        uniform_detection = request.form.get('uniform_detection') == 'on'
        id_card_detection = request.form.get('id_card_detection') == 'on'  
        shoes_detection = request.form.get('shoes_detection') == 'on'
        ciggerate_detection = request.form.get('ciggerate_detection') == 'on'
        sit_detection = request.form.get('sit_detection') == 'on'
        
        # Add other form fields from your model here, e.g.
        # gender_detection = request.form.get('gender_detection') == 'on'
        # face_detection = request.form.get('face_detection', 'on') == 'on' # Default if needed
        # ... and coordinates, cam_start, cam_stop etc.

        if not cam_name or not camera_type:
            return jsonify({'status': 'error', 'message': 'Camera name and type are required.'}), 400

        feed_src_to_store = None
        if camera_type == 'webcam':
            feed_src_to_store = '0'
        elif camera_type == 'cctv':
            if not feed_src_rtsp:
                return jsonify({'status': 'error', 'message': 'RTSP URL is required for CCTV type.'}), 400
            feed_src_to_store = feed_src_rtsp
        else:
            return jsonify({'status': 'error', 'message': 'Invalid camera type selected.'}), 400

        try:
            new_cam = Camera_Settings(
                cam_name=cam_name,
                feed_src=feed_src_to_store,
                role_camera=Camera_Settings.ROLE_TRACKING, # Ensure this constant exists
                company_id=company_obj.id, # Correctly use the ID of the fetched object
                cam_is_active=True,
                uniform_detection=uniform_detection,
                id_card_detection=id_card_detection,    
                shoes_detection=shoes_detection,
                ciggerate_detection=ciggerate_detection,
                sit_detection=sit_detection
                # ... add other fields to the constructor:
                # gender_detection=gender_detection,
                # face_detection=face_detection,
                # x1=request.form.get('x1', 0, type=int), ... etc.
            )
            db.session.add(new_cam)
            db.session.commit()
            
            # Return JSON response for successful AJAX POST
            return jsonify({
                'status': 'success',
                'message': 'Tracking camera added successfully!',
                'redirect_url': url_for('admin.tracking_cam') # Provide URL for client-side redirect
            }), 200 # HTTP 200 OK (or 201 Created)

        except Exception as e:
            db.session.rollback()
            # current_app.logger.error(f"Error adding camera: {e}", exc_info=True) # Good for server logs
            return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500 # Internal Server Error
            
    # For GET request, render the template as before
    return render_template('admin_panel/tracking_cam.html', company=company_obj)

@bp.route('/edit_tracking_camera/<int:cam_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_tracking_camera(cam_id):
    camera = Camera_Settings.query_get_or_404(cam_id) # Your actual query
    # Ensure company_obj and company_linked are actual attributes/relationships
    if not hasattr(camera, 'company_obj') or not hasattr(current_user, 'company_linked') or \
       camera.company_obj.id != current_user.company_linked.id: # Adjust .id if necessary
        flash("Unauthorized access to camera settings.", "danger")
        return redirect(url_for('admin.tracking_cam')) # Adjust to your camera list route

    if camera.role_camera != Camera_Settings.ROLE_TRACKING:
        flash("This is not a tracking camera.", "warning")
        return redirect(url_for('admin.tracking_cam')) # Adjust

    if request.method == 'POST':
        camera.cam_name = request.form.get('cam_name', camera.cam_name)
        camera.uniform_detection = request.form.get('uniform_detection') == 'on'
        camera.id_card_detection = request.form.get('id_card_detection') == 'on'
        camera.shoes_detection = request.form.get('shoes_detection') == 'on'
        camera.ciggerate_detection = request.form.get('ciggerate_detection') == 'on'
        camera.sit_detection = request.form.get('sit_detection') == 'on'
        camera.cam_is_active = request.form.get('cam_is_active') == 'on'

        new_camera_type = request.form.get('camera_type')
        new_feed_src_rtsp = request.form.get('feed_src_rtsp')
        new_feed_src_to_store = None
        valid_source_info = True

        if not new_camera_type:
            flash('Camera type is required.', 'danger')
            valid_source_info = False
        elif new_camera_type == 'webcam':
            new_feed_src_to_store = '0'
        elif new_camera_type == 'cctv':
            if not new_feed_src_rtsp:
                flash('RTSP URL is required for CCTV type.', 'danger')
                valid_source_info = False
            else:
                new_feed_src_to_store = new_feed_src_rtsp
        else:
            flash('Invalid camera type selected.', 'danger')
            valid_source_info = False
        
        if not valid_source_info:
            # Pass current type and RTSP value back to prefill form on error
            current_camera_type = 'webcam' if camera.feed_src == '0' else 'cctv'
            current_feed_src_rtsp = camera.feed_src if current_camera_type == 'cctv' else ''
            return render_template('admin_panel/edit_tracking_camera.html', 
                                   Active_Cam=camera, 
                                   form_data=request.form,
                                   current_camera_type=current_camera_type,
                                   current_feed_src_rtsp=current_feed_src_rtsp)

        if camera.feed_src != new_feed_src_to_store:
            if camera.feed_src: # If there was an old feed source
                release_camera_instance(camera.feed_src) 
            camera.feed_src = new_feed_src_to_store
        
        db.session.commit()
        flash('Tracking camera updated successfully!', 'success')
        return redirect(url_for('admin.tracking_cam')) # Adjust
    
    # For GET request
    current_camera_type = 'webcam' if camera.feed_src == '0' else 'cctv'
    current_feed_src_rtsp = camera.feed_src if current_camera_type == 'cctv' else ''
    return render_template('admin_panel/edit_tracking_camera.html', 
                           Active_Cam=camera,
                           current_camera_type=current_camera_type,
                           current_feed_src_rtsp=current_feed_src_rtsp)



@bp.route('/add_presence_camera', methods=['GET', 'POST'])
@login_required
@admin_required
def add_presence_camera():
    # Adjust this query based on your actual Company-User relationship
    # Option 1: If User has a direct company_id or company relationship
    # company = current_user.company 
    # Option 2: If Company has a user_account_id linking to User.id
    company = Company.query.filter_by(user_id=current_user.id).first() # Example, adjust!
    # Option 3: If Company model has a relationship like 'user_accounts' (many-to-many or one-to-many)
    # company = Company.query.filter(Company.user_accounts.any(id=current_user.id)).first()


    if not company:
        if request.method == 'POST': # AJAX request
            return jsonify({'status': 'error', 'message': 'Admin account not linked to a company.'}), 403
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login')) # Adjust if auth blueprint is different

    if request.method == 'POST':
        cam_name = request.form.get('cam_name')
        camera_type = request.form.get('camera_type') # IMPORTANT: Ensure your ADD modal sends this
        feed_src_rtsp = request.form.get('feed_src_rtsp') # IMPORTANT: Ensure your ADD modal sends this if CCTV
        attendance_time_start_str = request.form.get('attendance_time_start')
        attendance_time_end_str = request.form.get('attendance_time_end')
        leaving_time_start_str = request.form.get('leaving_time_start')
        leaving_time_end_str = request.form.get('leaving_time_end')

        required_fields = {
            'cam_name': cam_name,
            'camera_type': camera_type,
            'attendance_time_start': attendance_time_start_str,
            'attendance_time_end': attendance_time_end_str,
            'leaving_time_start': leaving_time_start_str,
            'leaving_time_end': leaving_time_end_str
        }

        missing_fields = [key for key, value in required_fields.items() if not value]
        if missing_fields:
            msg = f"Missing required fields: {', '.join(missing_fields)}. RTSP URL is also required if camera type is CCTV."
            return jsonify({'status': 'error', 'message': msg}), 400

        feed_src_to_store = None
        if camera_type == 'webcam':
            feed_src_to_store = '0' # Convention for webcam
        elif camera_type == 'cctv':
            if not feed_src_rtsp:
                return jsonify({'status': 'error', 'message': 'RTSP URL is required for CCTV type.'}), 400
            feed_src_to_store = feed_src_rtsp
        else:
            return jsonify({'status': 'error', 'message': 'Invalid camera type selected.'}), 400

        try:
            attendance_time_start = format_time(attendance_time_start_str).strftime('%H:%M:%S')
            attendance_time_end = format_time(attendance_time_end_str).strftime('%H:%M:%S')
            leaving_time_start = format_time(leaving_time_start_str).strftime('%H:%M:%S')
            leaving_time_end = format_time(leaving_time_end_str).strftime('%H:%M:%S')
        except ValueError as e:
            return jsonify({'status': 'error', 'message': f'Invalid time format. Please use HH:MM. Error: {e}'}), 400
        except AttributeError: # If format_time returns None or an object without strftime
             return jsonify({'status': 'error', 'message': f'Time formatting failed. Ensure time is HH:MM.'}), 400


        new_cam = Camera_Settings(
            cam_name=cam_name,
            feed_src=feed_src_to_store,
            role_camera=Camera_Settings.ROLE_PRESENCE, # Ensure this constant is defined in your model
            attendance_time_start=attendance_time_start,
            attendance_time_end=attendance_time_end,
            leaving_time_start=leaving_time_start,
            leaving_time_end=leaving_time_end,
            company_id=company.id,
            cam_is_active=True # Default to active
        )
        db.session.add(new_cam)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Presence camera added successfully!'}), 201

    # For GET request, render the page that potentially includes the "Add" modal
    # This route might not be directly for rendering a separate "add_presence_camera.html" page
    # if adding is always done via a modal on a list page.
    # If so, the main page's route (e.g., 'admin.presence_cam') should pass 'company'.
    return render_template('admin_panel/add_presence_camera.html', company=company, form_data={})


@bp.route('/edit_presence_camera/<int:cam_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_presence_camera(cam_id):
    camera = Camera_Settings.query.get_or_404(cam_id)

    # Authorization: Ensure the camera belongs to the current_user's company
    # Adjust based on your User and Company model relationships
    # Example: Assuming user has a 'company_linked' attribute that is a Company object
    # And Camera_Settings has a 'company' relationship attribute.
    if not hasattr(current_user, 'company_linked') or not current_user.company_linked:
        if request.method == 'POST':
             return jsonify({'status': 'error', 'message': "Admin account not linked to a company."}), 403
        flash("Admin account not linked to a company.", "danger")
        return redirect(url_for('auth.login')) # Adjust

    if camera.company_id != current_user.company_linked.id:
        if request.method == 'POST':
            return jsonify({'status': 'error', 'message': "Unauthorized access to camera settings."}), 403
        flash("Unauthorized access to camera settings.", "danger")
        return redirect(url_for('admin.presence_cam')) # Adjust (e.g., to your main camera list page)

    if camera.role_camera != Camera_Settings.ROLE_PRESENCE:
        if request.method == 'POST':
             return jsonify({'status': 'error', 'message': "This is not a presence camera."}), 400
        flash("This is not a presence camera.", "warning")
        return redirect(url_for('admin.presence_cam')) # Adjust

    if request.method == 'POST':
        camera.cam_name = request.form.get('cam_name', camera.cam_name)
        # IMPORTANT: Ensure your EDIT modal sends 'cam_is_active' (e.g. as a checkbox)
        camera.cam_is_active = request.form.get('cam_is_active') == 'on'

        try:
            time_fields_to_update = {
                'attendance_time_start': request.form.get('attendance_time_start'),
                'attendance_time_end': request.form.get('attendance_time_end'),
                'leaving_time_start': request.form.get('leaving_time_start'),
                'leaving_time_end': request.form.get('leaving_time_end'),
            }
            for field_name, time_str in time_fields_to_update.items():
                if time_str: # Only update if a new value is provided
                    setattr(camera, field_name, format_time(time_str).strftime('%H:%M:%S'))
        except ValueError as e:
            return jsonify({'status': 'error', 'message': f'Invalid time format. Please use HH:MM. Error: {e}'}), 400
        except AttributeError:
             return jsonify({'status': 'error', 'message': f'Time formatting failed. Ensure time is HH:MM.'}), 400


        # IMPORTANT: Ensure your EDIT modal allows changing camera_type and feed_src_rtsp if needed
        new_camera_type = request.form.get('camera_type')
        new_feed_src_rtsp = request.form.get('feed_src_rtsp')

        if new_camera_type: # If user is trying to update camera type/source
            new_feed_src_to_store = None
            if new_camera_type == 'webcam':
                new_feed_src_to_store = '0'
            elif new_camera_type == 'cctv':
                if not new_feed_src_rtsp:
                    return jsonify({'status': 'error', 'message': 'RTSP URL is required if changing to CCTV type.'}), 400
                new_feed_src_to_store = new_feed_src_rtsp
            else:
                return jsonify({'status': 'error', 'message': 'Invalid new camera type selected.'}), 400

            if camera.feed_src != new_feed_src_to_store:
                if camera.feed_src: # If there was an old feed source
                    release_camera_instance(camera.feed_src) # Assumed helper function
                camera.feed_src = new_feed_src_to_store
        
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Presence camera updated successfully!'}), 200

    # For GET request (populating the edit form/page if not solely relying on JS for modal population)
    # The provided JS `openEditModal` fetches data via 'camera.get_camera_data',
    # so this GET handler might be for a dedicated edit page.
    current_camera_type = 'webcam' if camera.feed_src == '0' else 'cctv'
    current_feed_src_rtsp = camera.feed_src if current_camera_type == 'cctv' else ''
    
    # Pass current form data for repopulation if rendering a full page
    form_data = {
        'cam_name': camera.cam_name,
        'cam_is_active': 'on' if camera.cam_is_active else '',
        'attendance_time_start': camera.attendance_time_start.strftime('%H:%M') if hasattr(camera.attendance_time_start, 'strftime') else camera.attendance_time_start, # Adjust if time is stored as string
        'attendance_time_end': camera.attendance_time_end.strftime('%H:%M') if hasattr(camera.attendance_time_end, 'strftime') else camera.attendance_time_end,
        'leaving_time_start': camera.leaving_time_start.strftime('%H:%M') if hasattr(camera.leaving_time_start, 'strftime') else camera.leaving_time_start,
        'leaving_time_end': camera.leaving_time_end.strftime('%H:%M') if hasattr(camera.leaving_time_end, 'strftime') else camera.leaving_time_end,
        'camera_type': current_camera_type,
        'feed_src_rtsp': current_feed_src_rtsp
    }
    return render_template('admin_panel/edit_presence_camera.html', 
                           Active_Cam=camera, 
                           form_data=form_data, # Send current camera data as form_data
                           current_camera_type=current_camera_type,
                           current_feed_src_rtsp=current_feed_src_rtsp)


@bp.route('/delete_camera/<int:cam_id>', methods=['POST'])
@login_required
@admin_required
def delete_camera(cam_id):
    camera = Camera_Settings.query.get_or_404(cam_id)

    # Authorization (similar to edit)
    if not hasattr(current_user, 'company_linked') or not current_user.company_linked:
        return jsonify({'status': 'error', 'message': "Admin account not linked to a company."}), 403
    if camera.company_id != current_user.company_linked.id:
        return jsonify({'status': 'error', 'message': "Unauthorized: Camera does not belong to your company."}), 403

    try:
        # Optional: Release camera instance if applicable before deleting
        if camera.feed_src:
            release_camera_instance(camera.feed_src)

        db.session.delete(camera)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Camera deleted successfully.'}), 200
    except Exception as e:
        db.session.rollback()
        # Log the error e
        return jsonify({'status': 'error', 'message': f'Error deleting camera: {str(e)}'}), 500

# It's assumed you have a route like this for populating the edit modal via JS
# This is what `openEditModal` in your JS seems to call.
@bp.route('/get_presence_camera_data/<int:cam_id>', methods=['GET'])
@login_required
@admin_required
def get_presence_camera_data(cam_id):
    camera = Camera_Settings.query.get_or_404(cam_id)
    # Authorization
    if not hasattr(current_user, 'company_linked') or not current_user.company_linked or camera.company_id != current_user.company_linked.id:
        return jsonify({'status': 'error', 'message': "Unauthorized"}), 403

    if camera.role_camera != Camera_Settings.ROLE_PRESENCE:
         return jsonify({'status': 'error', 'message': "Not a presence camera."}), 400

    # Convert time objects to "HH:MM" strings for form input type="time"
    # Your database might store times as strings or Python time objects. Adjust as needed.
    def format_db_time_to_hhmm(db_time):
        if hasattr(db_time, 'strftime'): # If it's a time/datetime object
            return db_time.strftime('%H:%M')
        if isinstance(db_time, str) and len(db_time) >= 5: # If it's a string like "HH:MM:SS"
            return db_time[:5]
        return "" # Default or handle error

    camera_data = {
        'id': camera.id,
        'cam_name': camera.cam_name,
        'feed_src': camera.feed_src,
        'role_camera': camera.role_camera,
        'attendance_time_start': format_db_time_to_hhmm(camera.attendance_time_start),
        'attendance_time_end': format_db_time_to_hhmm(camera.attendance_time_end),
        'leaving_time_start': format_db_time_to_hhmm(camera.leaving_time_start),
        'leaving_time_end': format_db_time_to_hhmm(camera.leaving_time_end),
        'cam_is_active': camera.cam_is_active,
        # Determine current_camera_type and current_feed_src_rtsp for the form
        'current_camera_type': 'webcam' if camera.feed_src == '0' else 'cctv',
        'current_feed_src_rtsp': camera.feed_src if camera.feed_src != '0' else ''
    }
    return jsonify({'status': 'success', 'data': camera_data})
