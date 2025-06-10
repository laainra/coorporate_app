# coorporate_app/app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
import os # Import os untuk mengakses variabel lingkungan
from werkzeug.security import generate_password_hash # Import untuk hashing password

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
migrate = Migrate()

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('config.Config')
    app.config.from_pyfile('config.py', silent=True)

    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
     # Import blueprint setelah app dibuat




    # Import blueprints (pastikan nama blueprint dan url_prefix sesuai)
    from app.routes.auth_routes import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/')

    from app.routes.superadmin_routes import bp as superadmin_bp
    app.register_blueprint(superadmin_bp, url_prefix='/superadmin')

    from app.routes.admin_routes import bp as admin_bp
    app.register_blueprint(admin_bp, url_prefix='/admin')

    from app.routes.employee_routes import bp as employee_bp
    app.register_blueprint(employee_bp, url_prefix='/employee')

    from app.routes.camera_routes import bp as camera_bp
    app.register_blueprint(camera_bp, url_prefix='/camera')

    from app.routes.personnel_routes import bp as personnel_bp
    app.register_blueprint(personnel_bp, url_prefix='/personnels')

    from app.routes.face_routes import bp as face_bp
    app.register_blueprint(face_bp, url_prefix='/face')

    from app.routes.stream_routes import bp as stream_bp
    app.register_blueprint(stream_bp, url_prefix='/stream')

    from app.routes.settings_routes import bp as settings_bp
    app.register_blueprint(settings_bp, url_prefix='/settings')

    # User loader for Flask-Login
    from app.models import Users
    @login_manager.user_loader
    def load_user(user_id):
        return Users.query.get(int(user_id))

    # ====================================================================
    # Logika Pembuatan Superadmin Otomatis
    # ====================================================================
    with app.app_context():
        from .routes import face_routes
        face_routes.initialize_ai_models(app)
        from app.models import Users # Import model di dalam context
        if not Users.query.filter_by(role=Users.ROLE_SUPERADMIN).first():
            # Jika belum ada superadmin, buat satu
            username = os.environ.get('SUPERADMIN_USERNAME', 'superadmin')
            email = os.environ.get('SUPERADMIN_EMAIL', 'superadmin@gmail.com')
            password = os.environ.get('SUPERADMIN_PASSWORD', 'superadmin') # Ganti 'superadminpass' dengan password yang lebih kuat
            
            # Periksa apakah username sudah ada (walaupun tidak superadmin)
            if not Users.query.filter_by(username=username).first():
                try:
                    new_superadmin = Users(
                        username=username,
                        email=email,
                        role=Users.ROLE_SUPERADMIN,
                    )
                    new_superadmin.set_password(password)
                    db.session.add(new_superadmin)
                    db.session.commit()
                    print(f"INFO: Superadmin user '{username}' created automatically.")
                except Exception as e:
                    db.session.rollback()
                    print(f"ERROR: Failed to create superadmin user: {e}")
            else:
                print(f"INFO: Superadmin username '{username}' already exists (not necessarily a superadmin role). Skipping creation.")
        else:
            print("INFO: Superadmin user already exists. Skipping automatic creation.")
    # ====================================================================

    return app

# Import semua model agar SQLAlchemy dapat mendeteksinya untuk migrasi
import app.models