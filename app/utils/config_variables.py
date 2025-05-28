import os

def get_base_dir():
    # BASE_DIR mengacu ke root project (coorporate_app)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def get_personnel_folder_path():
    # Path ke folder foto personel, misal: coorporate_app/app/static/img/personnel_pics
    return os.path.join(get_base_dir(), 'app', 'static', 'img', 'personnel_pics')

def get_report_folder_path():
    # Path ke folder report, misal: coorporate_app/app/static
    return os.path.join(get_base_dir(), 'app', 'static')

def get_presence_folder_path():
    # Path ke folder extracted_faces, misal: coorporate_app/app/static/img/extracted_faces
    return os.path.join(get_base_dir(), 'app', 'static', 'img', 'extracted_faces')