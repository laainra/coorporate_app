# coorporate_app/config.py
import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'bismillah-ta'
    
    # Konfigurasi database MySQL
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:@localhost/cctv_db' 
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Konfigurasi untuk file upload
    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'static','img') 
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 # 16 MB max upload size

    # Path untuk AI/CV assets (diletakkan di static)
    HAARCASCADES_PATH = os.path.join(basedir, 'app', 'static', 'haarcascades')
    TRAINED_MODELS_PATH = os.path.join(basedir, 'app', 'static', 'trained_models')
    YOLO_MODELS_PATH = os.path.join(basedir, 'app', 'static', 'yolo')

    # Flask-Login configuration
    REMEMBER_COOKIE_DURATION = timedelta(days=7) # Example: session lasts for 7 days
    PERMANENT_SESSION_LIFETIME = timedelta(days=30) # Example: session timeout for 30 minutes of inactivity