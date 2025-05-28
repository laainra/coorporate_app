# coorporate_app/app/utils/helpers.py
import os
from werkzeug.utils import secure_filename
from flask import current_app

def allowed_file(filename, extensions=None):
    if extensions is None:
        extensions = {'png', 'jpg', 'jpeg', 'gif', 'xlsx', 'xls'} # Default extensions
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions

def save_uploaded_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filename
    return None

# Anda bisa menambahkan fungsi helper lain di sini, misal untuk excel export, dll.