# coorporate_app/app/models.py
from app import db # Import db instance from the main app
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# ====================================================================
# User, Company, Division, and Personnel Models
# ====================================================================

class Users(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    
    ROLE_SUPERADMIN = 'superadmin'
    ROLE_ADMIN = 'admin'
    ROLE_EMPLOYEE = 'employee'
    ROLE_CHOICES = [
        (ROLE_SUPERADMIN, 'Super Admin'),
        (ROLE_ADMIN, 'Admin'),
        (ROLE_EMPLOYEE, 'Employee'),
    ]
    role = db.Column(db.String(20), default=ROLE_EMPLOYEE, nullable=False)
    
    email = db.Column(db.String(120), unique=True, nullable=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False) # Store hashed password
    createdAt = db.Column(db.DateTime, default=datetime.now)
    updatedAt = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    @property
    def is_superadmin(self):
        return self.role == self.ROLE_SUPERADMIN

    @property
    def is_admin(self):
        return self.role == self.ROLE_ADMIN

    @property
    def is_employee(self):
        return self.role == self.ROLE_EMPLOYEE

    # Relationships (adjust backref names to avoid conflicts if they clash with column names)
    company_linked = db.relationship('Company', backref='user_account', uselist=False, lazy=True)
    personnel_linked = db.relationship('Personnels', backref='user_account', uselist=False, lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # UserMixin required methods
    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f'<User {self.username}>'

class Company(db.Model):
    __tablename__ = 'company'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    createdAt = db.Column(db.DateTime, default=datetime.now)
    updatedAt = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), unique=True, nullable=True) # OneToOneField in Django

    divisions = db.relationship('Divisions', backref='company_obj', lazy='dynamic', cascade="all, delete-orphan")
    personnels = db.relationship('Personnels', backref='company_obj', lazy='dynamic', cascade="all, delete-orphan")
    camera_settings = db.relationship('Camera_Settings', backref='company_obj', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Company {self.name}>'

class Divisions(db.Model):
    __tablename__ = 'divisions'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id', ondelete='CASCADE'), nullable=False) # ForeignKey in Django
    createdAt = db.Column(db.DateTime, default=datetime.now)
    updatedAt = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    personnels = db.relationship('Personnels', backref='division_obj', lazy='dynamic')

    def __repr__(self):
        return f'<Division {self.name}>'

class Personnels(db.Model):
    __tablename__ = 'personnels'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    # employeeid = db.Column(db.String(100), unique=True, nullable=True) # Un-comment if needed
    # gender = db.Column(db.String(10)) # Un-comment if needed
    # employment_status = db.Column(db.String(100)) # Un-comment if needed

    createdAt = db.Column(db.DateTime, default=datetime.now)
    updatedAt = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), unique=True, nullable=True) # OneToOneField in Django
    division_id = db.Column(db.Integer, db.ForeignKey('divisions.id', ondelete='SET NULL'), nullable=True) # ForeignKey in Django
    company_id = db.Column(db.Integer, db.ForeignKey('company.id', ondelete='CASCADE'), nullable=True) # ForeignKey in Django

    images = db.relationship('Personnel_Images', backref='personnel_obj', lazy='dynamic', cascade="all, delete-orphan")
    entries = db.relationship('Personnel_Entries', backref='personnel_obj', lazy='dynamic', cascade="all, delete-orphan")
    work_timers = db.relationship('Work_Timer', backref='personnel_obj', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Personnel {self.name}>'

# ====================================================================
# Camera Settings and Counted Instances Models
# ====================================================================

class Camera_Settings(db.Model):
    __tablename__ = 'camera_settings'
    id = db.Column(db.Integer, primary_key=True)
    cam_name = db.Column(db.String(200), nullable=False)
    feed_src = db.Column(db.String(200), nullable=True, default="0")
    
    # Coordinates for detection area
    x1 = db.Column(db.Integer, default=0)
    y1 = db.Column(db.Integer, default=0)
    x2 = db.Column(db.Integer, default=0)
    y2 = db.Column(db.Integer, default=0)
    x3 = db.Column(db.Integer, default=0)
    y3 = db.Column(db.Integer, default=0)
    x4 = db.Column(db.Integer, default=0)
    y4 = db.Column(db.Integer, default=0)
    x5 = db.Column(db.Integer, default=0)
    y5 = db.Column(db.Integer, default=0)
    x6 = db.Column(db.Integer, default=0)
    y6 = db.Column(db.Integer, default=0)
    x7 = db.Column(db.Integer, default=0)
    y7 = db.Column(db.Integer, default=0)
    x8 = db.Column(db.Integer, default=0)
    y8 = db.Column(db.Integer, default=0)
    
    cam_is_active = db.Column(db.Boolean, default=True)
    gender_detection = db.Column(db.Boolean, default=False)
    face_detection = db.Column(db.Boolean, default=True)
    face_capture = db.Column(db.Boolean, default=True)
    id_card_detection = db.Column(db.Boolean, default=False)
    uniform_detection = db.Column(db.Boolean, default=False)
    shoes_detection = db.Column(db.Boolean, default=False)
    ciggerate_detection = db.Column(db.Boolean, default=False)
    sit_detection = db.Column(db.Boolean, default=False)
    
    cam_start = db.Column(db.String(200), default="00:01:00", nullable=True) # Store as string "HH:MM:SS"
    cam_stop = db.Column(db.String(200), default="23:59:00", nullable=True)
    attendance_time_start = db.Column(db.String(200), nullable=True)
    attendance_time_end = db.Column(db.String(200), nullable=True)
    leaving_time_start = db.Column(db.String(200), nullable=True)
    leaving_time_end = db.Column(db.String(200), nullable=True)
    
    createdAt = db.Column(db.DateTime, default=datetime.now)
    updatedAt = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    ROLE_PRESENCE = 'P'
    ROLE_TRACKING = 'T'
    ROLE_CAMERA_CHOICES = [
        (ROLE_PRESENCE, 'Presence'),
        (ROLE_TRACKING, 'Tracking')
    ]
    role_camera = db.Column(db.String(10), default=ROLE_TRACKING, nullable=False)
    
    company_id = db.Column(db.Integer, db.ForeignKey('company.id', ondelete='CASCADE'), nullable=True) # ForeignKey in Django

    counted_instances = db.relationship('Counted_Instances', backref='camera_obj', lazy='dynamic', cascade="all, delete-orphan")
    personnel_entries = db.relationship('Personnel_Entries', backref='camera_obj', lazy='dynamic') # cascade might be handled by Personnel_Entries itself
    work_timers = db.relationship('Work_Timer', backref='camera_obj', lazy='dynamic')

    def __repr__(self):
        return f'<Camera_Settings {self.cam_name}>'

class Counted_Instances(db.Model):
    __tablename__ = 'counted_instances'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera_settings.id', ondelete='CASCADE'), nullable=False) # ForeignKey in Django
    
    male_entries = db.Column(db.Integer, default=0)
    female_entries = db.Column(db.Integer, default=0)
    unknown_gender_entries = db.Column(db.Integer, default=0)
    staff_entries = db.Column(db.Integer, default=0)
    intern_entries = db.Column(db.Integer, default=0)
    unknown_role_entries = db.Column(db.Integer, default=0)
    people_exits = db.Column(db.Integer, default=0)
    people_inside = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f'<Counted_Instances {self.timestamp} - Cam:{self.camera_id}>'

# ====================================================================
# Presence and Work Timer Models
# ====================================================================

class Personnel_Images(db.Model):
    __tablename__ = 'personnel_images'
    id = db.Column(db.Integer, primary_key=True)
    personnel_id = db.Column(db.Integer, db.ForeignKey('personnels.id', ondelete='CASCADE'), nullable=False) # ForeignKey in Django
    image_path = db.Column(db.String(255), nullable=True) # Path/lokasi file gambar.
    createdAt = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f'<Personnel_Image {self.image_path} - Personnel:{self.personnel_id}>'

class Personnel_Entries(db.Model):
    __tablename__ = 'personnel_entries'
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera_settings.id', ondelete='CASCADE'), nullable=False) # ForeignKey in Django
    timestamp = db.Column(db.DateTime, nullable=False)
    
    STATUS_ONTIME = 'ONTIME'
    STATUS_LATE = 'LATE'
    STATUS_LEAVE = 'LEAVE'
    STATUS_UNKNOWN = 'UNKNOWN'
    status_choices = [
        (STATUS_ONTIME, 'On Time'),
        (STATUS_LATE, 'Late'),
        (STATUS_LEAVE, 'Already Leave'),
        (STATUS_UNKNOWN, 'No Presence'),
    ]
    presence_status = db.Column(db.String(10), default=STATUS_UNKNOWN, nullable=False)
    
    personnel_id = db.Column(db.Integer, db.ForeignKey('personnels.id', ondelete='CASCADE'), nullable=False) # ForeignKey in Django
    image = db.Column(db.String(255), nullable=True) # Path/lokasi gambar presensi.

    def __repr__(self):
        return f'<Personnel_Entry {self.timestamp} - {self.personnel_id} - {self.presence_status}>'
    
class Work_Timer(db.Model):
    __tablename__ = 'work_timer'
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera_settings.id', ondelete='SET NULL'), nullable=True) # ForeignKey in Django
    datetime = db.Column(db.DateTime, nullable=False)
    
    TYPE_SIT = 'SIT'
    TYPE_FACE_DETECTED = 'FACE_DETECTED'
    type_choices = [
        (TYPE_SIT, 'Sit'),
        (TYPE_FACE_DETECTED, 'Face Detected'),
    ]
    type = db.Column(db.String(15), nullable=False) # Jenis timer (Sit, Face Detected).
    
    timer = db.Column(db.Integer, default=0, nullable=False) # Durasi timer.
    personnel_id = db.Column(db.Integer, db.ForeignKey('personnels.id', ondelete='CASCADE'), nullable=False) # ForeignKey in Django

    def __repr__(self):
        return f'<Work_Timer {self.datetime} - {self.personnel_id} - {self.type}>'

class Tracking(db.Model):
    __tablename__ = 'tracking'
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera_settings.id', ondelete='CASCADE'), nullable=False)
    detected_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    image_path = db.Column(db.String(255), nullable=True)
    personnel_id = db.Column(db.Integer, db.ForeignKey('personnels.id', ondelete='SET NULL'), nullable=True)  # <--- Tambahkan ini
    # Optional: personnel_name = db.Column(db.String(100), nullable=True)

    camera = db.relationship('Camera_Settings', backref='trackings')
    personnel = db.relationship('Personnels', backref='trackings')  # <--- Tambahkan ini jika ingin akses relasi

    def __repr__(self):
        return f"<Tracking {self.detected_class} - {self.timestamp}>"