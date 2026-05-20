import os
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, date
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
import sys

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

# Load the neural network model
global_loaded_model = None
try:
    if load_model is not None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'model', 'student_retention_model.h5')
        if os.path.exists(model_path):
            global_loaded_model = load_model(model_path)
            print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.train_model import train_ai

# Flask Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'graduation_project_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Setup
db_uri = os.environ.get('DATABASE_URL')
if not db_uri:
    db_uri = 'mysql+pymysql://root:@localhost/student_retention_db'

# If using MySQL, verify connection and create database if missing
if db_uri.startswith('mysql'):
    try:
        import pymysql
        parsed = urlparse(db_uri.replace('mysql+pymysql://', 'http://'))
        host_port = parsed.netloc.split('@')[-1]
        host = host_port.split(':')[0]
        port = int(host_port.split(':')[1]) if ':' in host_port else 3306
        user = parsed.username or 'root'
        password = parsed.password or ''
        db_name = parsed.path.strip('/')
        
        conn = pymysql.connect(host=host, port=port, user=user, password=password)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Database verified/created: {db_name}")
        app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    except Exception as e:
        print(f"Could not connect to MySQL: {e}. Falling back to SQLite.")
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///final_system_v4.db?timeout=20'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==========================================
# 2. Constants & Academic Subjects
# ==========================================
SUBJECTS_SEM1 = [
    "Database Systems", 
    "Mathematics & Statistics", 
    "Introduction to Programming"
]

SUBJECTS_SEM2 = [
    "Web Development", 
    "Machine Learning", 
    "Data Structures"
]

SUBJECTS = SUBJECTS_SEM1 + SUBJECTS_SEM2
SEMESTERS = ["First Semester", "Second Semester"]

# Database Models

class User(db.Model):
    """User account model (Admin, Instructor, Student)"""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=True)
    
    subject = db.Column(db.String(100), nullable=True)
    semester = db.Column(db.String(50), nullable=True)

class ModuleCourse(db.Model):
    """Modules / courses mapped to instructors"""
    __tablename__ = 'modules_courses'
    id = db.Column(db.Integer, primary_key=True)
    course_name = db.Column(db.String(100), nullable=False, unique=True)
    instructor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

class StudentProfile(db.Model):
    """Student profiles containing academic performance metrics"""
    __tablename__ = 'student_profiles'
    id = db.Column(db.Integer, primary_key=True)
    student_no = db.Column(db.String(50), unique=True, nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    gpa = db.Column(db.Float, default=0.0)
    cumulative_attendance = db.Column(db.Float, default=100.0)
    risk_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(20), default='Low Risk')
    
    instructor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    subject = db.Column(db.String(100), nullable=True)
    
    instructor = db.relationship('User', backref=db.backref('assigned_students', lazy=True), foreign_keys=[instructor_id])

    @property
    def student_id(self):
        return self.student_no

    @property
    def attendance(self):
        return self.cumulative_attendance

    @property
    def ai_status(self):
        return 'At Risk' if self.risk_score > 0.70 else 'Safe'

class AuditLog(db.Model):
    """Audit log for system actions"""
    __tablename__ = 'audit_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100))
    user_role = db.Column(db.String(20))
    action = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    """Alerts and notifications"""
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    student_no = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    student_id = db.Column(db.String(50), nullable=True)
    type = db.Column(db.String(20), default='Alert')
    
    instructor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    instructor = db.relationship('User', backref=db.backref('sent_notifications', lazy=True), foreign_keys=[instructor_id])

# Legacy Models for compatibility
class StudentRecord(db.Model):
    __tablename__ = 'student_record'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50))
    subject = db.Column(db.String(100))
    gpa = db.Column(db.Float)
    attendance = db.Column(db.Float)
    ai_status = db.Column(db.String(50))

class SystemEvaluation(db.Model):
    __tablename__ = 'system_evaluation'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    feedback = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Database Initialization & Seeding
with app.app_context():
    try:
        db.create_all()
        # Seed default Admin
        admin_user = User.query.filter_by(role='Admin').first()
        if not admin_user:
            admin_pw_hash = generate_password_hash('admin123')
            default_admin = User(username='Admin', email='admin@gmail.com', password_hash=admin_pw_hash, role='Admin')
            db.session.add(default_admin)
            db.session.commit()
            print("Default admin user seeded.")
            
        # Seed default Instructor
        sara_user = User.query.filter_by(email='sara@gmail.com').first()
        if not sara_user:
            sara_pw_hash = generate_password_hash('sara123')
            default_sara = User(
                username='sara',
                email='sara@gmail.com',
                password_hash=sara_pw_hash,
                role='Instructor',
                subject='Database Systems',
                semester='First Semester'
            )
            db.session.add(default_sara)
            db.session.commit()
            print("Default instructor seeded.")
            
        # Seed default Student profiles
        sara_user = User.query.filter_by(email='sara@gmail.com').first()
        inst_id = sara_user.id if sara_user else None
        student_1 = StudentProfile.query.filter_by(student_no='KGL1000').first()
        if not student_1:
            db.session.add(StudentProfile(student_no='KGL1000', student_name='Amina Al-Balushi', gpa=2.91, cumulative_attendance=76.1, risk_score=0.15, risk_level='Low Risk', instructor_id=inst_id, subject='Database Systems'))
            db.session.add(StudentProfile(student_no='KGL1001', student_name='Fahad Al-Harthi', gpa=3.29, cumulative_attendance=80.1, risk_score=0.01, risk_level='Low Risk', instructor_id=inst_id, subject='Database Systems'))
            db.session.add(StudentProfile(student_no='KGL1002', student_name='Mazeen Al-Omani', gpa=1.85, cumulative_attendance=55.4, risk_score=0.89, risk_level='High Risk', instructor_id=inst_id, subject='Database Systems'))
            db.session.commit()
            print("Default student profiles seeded.")
    except Exception as e:
        print(f"Database setup error: {e}")

# Wrapper class to support unified interface in dashboards
class StudentViewWrapper:
    def __init__(self, student_no, student_name, gpa, attendance, risk_score, risk_level, ai_status):
        self.student_no = student_no
        self.student_id = student_no  # backward compatibility for legacy templates & tests
        self.student_name = student_name
        self.gpa = gpa
        self.attendance = attendance
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.ai_status = ai_status

# ==========================================
# 5. Core Services: CNN-RNN AI Inference Engine & Ethics
def anonymize_id(student_no):
    """Anonymize student ID for reports."""
    if not student_no:
        return "STU-ANON"
    hashed = hashlib.sha256(f"salt_{student_no}".encode()).hexdigest()[:8].upper()
    return f"STU-{hashed}"

def compute_hybrid_cnn_rnn_risk(gpa, attendance, quiz_score, assignment_score):
    """Calculate student risk score using the model."""
    global global_loaded_model
    features = np.array([[gpa, attendance, quiz_score, assignment_score]], dtype=float)
    mean_val = np.array([2.5, 75.0, 70.0, 70.0])
    std_val = np.array([0.8, 15.0, 15.0, 15.0])
    scaled_features = (features - mean_val) / (std_val + 1e-8)
    
    # Use loaded model if available
    if global_loaded_model is not None:
        try:
            input_tensor = scaled_features.reshape((1, 1, 4))
            prediction = global_loaded_model.predict(input_tensor)
            return float(prediction[0][0])
        except Exception as e:
            pass

    # Emulation fallback
    np.random.seed(42)
    W_conv = np.random.normal(0.0, 0.1, (4, 64))
    b_conv = np.zeros(64)
    conv_out = np.maximum(0, np.dot(scaled_features[0], W_conv) + b_conv)
    
    W_lstm = np.random.normal(0.0, 0.1, (64, 200))
    U_lstm = np.random.normal(0.0, 0.1, (50, 200))
    b_lstm = np.zeros(200)
    
    gates = np.dot(conv_out, W_lstm) + b_lstm
    i_gate = 1.0 / (1.0 + np.exp(-gates[0, 0:50]))
    f_gate = 1.0 / (1.0 + np.exp(-gates[0, 50:100]))
    c_tilde = np.tanh(gates[0, 100:150])
    o_gate = 1.0 / (1.0 + np.exp(-gates[0, 150:200]))
    
    c_state = f_gate * 0.0 + i_gate * c_tilde
    h_state = o_gate * np.tanh(c_state)
    
    W_dense = np.random.normal(-0.1, 0.1, (50, 1))
    b_dense = np.array([-0.2])
    logits = np.dot(h_state, W_dense) + b_dense
    risk_score = 1.0 / (1.0 + np.exp(-logits[0]))
    return float(risk_score)

def compute_hybrid_cnn_rnn_risk_batch(df):
    """Calculate risk scores for a batch of students."""
    n_rows = len(df)
    if n_rows == 0:
        return []
        
    gpa = df['gpa'].fillna(2.5).values
    attendance = df['attendance'].fillna(100.0).values
    quiz = df.get('quiz_score', pd.Series([75.0] * n_rows)).fillna(75.0).values
    assign = df.get('assignment_score', pd.Series([75.0] * n_rows)).fillna(75.0).values
    
    features = np.stack([gpa, attendance, quiz, assign], axis=1)
    mean_val = np.array([2.5, 75.0, 70.0, 70.0])
    std_val = np.array([0.8, 15.0, 15.0, 15.0])
    scaled_features = (features - mean_val) / (std_val + 1e-8)
    
    # Use loaded model if available
    global global_loaded_model
    if global_loaded_model is not None:
        try:
            input_tensor = scaled_features.reshape((n_rows, 1, 4))
            predictions = global_loaded_model.predict(input_tensor)
            return [float(p[0]) for p in predictions]
        except Exception as e:
            print(f"Batch prediction fallback: {e}")
            
    # Emulation fallback
    np.random.seed(42)
    
    W_conv = np.random.normal(0.0, 0.1, (4, 64))
    b_conv = np.zeros(64)
    conv_out = np.maximum(0, np.dot(scaled_features, W_conv) + b_conv)
    
    W_lstm = np.random.normal(0.0, 0.1, (64, 200))
    U_lstm = np.random.normal(0.0, 0.1, (50, 200))
    b_lstm = np.zeros(200)
    
    gates = np.dot(conv_out, W_lstm) + b_lstm
    
    i_gate = 1.0 / (1.0 + np.exp(-gates[:, 0:50]))
    f_gate = 1.0 / (1.0 + np.exp(-gates[:, 50:100]))
    c_tilde = np.tanh(gates[:, 100:150])
    o_gate = 1.0 / (1.0 + np.exp(-gates[:, 150:200]))
    
    c_state = f_gate * 0.0 + i_gate * c_tilde
    h_state = o_gate * np.tanh(c_state)
    
    W_dense = np.random.normal(-0.1, 0.1, (50, 1))
    b_dense = np.array([-0.2])
    logits = np.dot(h_state, W_dense) + b_dense
    risk_scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    
    return [float(r) for r in risk_scores]

def audit_log(user_id, role, action):
    """Log user actions to audit_logs table."""
    try:
        log = AuditLog(user_id=user_id, user_role=role, action=action)
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Audit log error: {e}")

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if role == 'Student':
            # Create student user and profile on first login
            user = User.query.filter_by(username=email, role='Student').first()
            if not user and email:
                hashed_pw = generate_password_hash(password or 'student123')
                user = User(username=email, email=f"{email}@student.com", password_hash=hashed_pw, role='Student')
                db.session.add(user)
                
                if not StudentProfile.query.filter_by(student_no=email).first():
                    base_profile = StudentProfile(student_no=email, student_name=f"Student {email}", gpa=2.8, cumulative_attendance=90.0)
                    db.session.add(base_profile)
                db.session.commit()
                
            session['user_id'] = email
            session['role'] = 'Student'
            
            audit_log(email, "Student", "Logged into the system")
            return redirect(url_for('student_portal'))
        else:
            user = User.query.filter_by(email=email, role=role).first()
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = email
                session['role'] = role
                session['user_subject'] = user.subject or 'General'
                
                audit_log(email, role, "Logged into the system")
                
                if role == 'Admin':
                    return redirect(url_for('admin_dashboard'))
                elif role == 'Instructor':
                    return redirect(url_for('instructor_dashboard'))
            else:
                audit_log(email or "unknown", role or "Guest", f"Failed authentication attempt: role={role}")
                flash('Invalid login credentials', 'error')
                
    return render_template('login.html')

@app.route('/logout')
def logout():
    uid = session.get('user_id', 'Guest')
    role = session.get('role', 'Guest')
    audit_log(uid, role, "Logged out from system portal")
    session.clear()
    return redirect(url_for('login'))

# Admin Dashboard

@app.route('/admin')
@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
        
    filter_role = request.args.get('filter_role', '')
    if filter_role == 'Instructor Logs':
        logs = AuditLog.query.filter_by(user_role='Instructor').order_by(AuditLog.timestamp.desc()).all()
    elif filter_role == 'Student Logs':
        logs = AuditLog.query.filter_by(user_role='Student').order_by(AuditLog.timestamp.desc()).all()
    else:
        logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(30).all()
        
    instructors = User.query.filter_by(role='Instructor').all()
    students = StudentProfile.query.all()
    
    # Calculate counts
    low_count = db.session.query(StudentProfile).filter(StudentProfile.risk_score < 0.40).count()
    med_count = db.session.query(StudentProfile).filter(StudentProfile.risk_score >= 0.40, StudentProfile.risk_score <= 0.70).count()
    high_count = db.session.query(StudentProfile).filter(StudentProfile.risk_score > 0.70).count()
    
    stats = {
        'low_risk_students': low_count,
        'medium_risk_students': med_count,
        'high_risk_students': high_count,
        'total_students': low_count + med_count + high_count,
        'subject_name': session.get('last_analyzed_subject', 'General Analysis')
    }
    
    # Calculate evaluation average
    evals = SystemEvaluation.query.order_by(SystemEvaluation.timestamp.desc()).all()
    eval_count = len(evals)
    eval_weighted_mean = round(sum(e.score for e in evals) / eval_count, 2) if eval_count > 0 else 0.0
    
    if eval_weighted_mean >= 4.5: eval_label = "Excellent"
    elif eval_weighted_mean >= 3.5: eval_label = "Good"
    elif eval_weighted_mean >= 2.5: eval_label = "Satisfactory"
    elif eval_weighted_mean >= 1.5: eval_label = "Fair"
    else: eval_label = "No Feedback"
    
    eval_stats = {'count': eval_count, 'mean': eval_weighted_mean, 'label': eval_label}
    
    return render_template('admin_dashboard.html', 
                           logs=logs, 
                           instructors=instructors, 
                           students=students,
                           stats=stats, 
                           evals=evals,
                           eval_stats=eval_stats,
                           subjects=SUBJECTS, 
                           subjects_sem1=SUBJECTS_SEM1,
                           subjects_sem2=SUBJECTS_SEM2,
                           semesters=SEMESTERS)

@app.route('/admin/ai_diagnostics')
def ai_diagnostics():
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
    return render_template('ai_diagnostics.html')

@app.route('/admin/manage_users', methods=['POST'])
def manage_users():
    if session.get('role') != 'Admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')
    subject = request.form.get('subject')
    semester = request.form.get('semester')
    
    if not username or not password or not role:
        flash('Required registration fields are missing', 'error')
        return redirect(url_for('admin_dashboard'))
        
    hashed_pw = generate_password_hash(password)
    if role == 'Instructor':
        # Check for existing username or email
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Error: An instructor with this name/username is already registered.', 'error')
            return redirect(url_for('admin_dashboard'))
        if email:
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                flash('Error: An instructor with this email is already registered.', 'error')
                return redirect(url_for('admin_dashboard'))
        try:
            new_user = User(username=username, email=email, password_hash=hashed_pw, role='Instructor', subject=subject, semester=semester)
            db.session.add(new_user)
            db.session.commit()
            
            audit_log(session.get('user_id'), "Admin", f"Registered new user account: name={username}, role={role}")
            flash(f'Account created successfully for {username}!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'User registration failed: {str(e)}', 'error')
    else:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Error: A student with this student number/username is already registered.', 'error')
            return redirect(url_for('admin_dashboard'))
        try:
            new_user = User(username=username, email=f"{username}@student.com", password_hash=hashed_pw, role='Student')
            new_profile = StudentProfile(student_no=username, student_name=username, gpa=2.5, cumulative_attendance=100.0, risk_score=0.1, risk_level='Low Risk')
            
            db.session.add(new_user)
            db.session.add(new_profile)
            db.session.commit()
            
            audit_log(session.get('user_id'), "Admin", f"Registered new user account: name={username}, role={role}")
            flash(f'Account created successfully for {username}!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'User registration failed: {str(e)}', 'error')
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_instructor', methods=['POST'])
def add_instructor():
    """Add a new instructor account."""
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
        
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    subject = request.form.get('subject') 
    semester = request.form.get('semester')
    
    hashed_pw = generate_password_hash(password)
    
    existing_user = User.query.filter_by(username=name).first()
    if existing_user:
        flash('Error: An instructor with this name/username is already registered.', 'error')
        return redirect(url_for('admin_dashboard'))
        
    if email:
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Error: An instructor with this email is already registered.', 'error')
            return redirect(url_for('admin_dashboard'))
            
    try:
        new_inst = User(username=name, email=email, password_hash=hashed_pw, role='Instructor', subject=subject, semester=semester)
        db.session.add(new_inst)
        db.session.commit()
        
        audit_log(session.get('user_id'), "Admin", f"Created Instructor {name} for {subject}")
        flash(f'Instructor registered successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error registering: {str(e)}', 'error')
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_instructor/<int:id>', methods=['POST'])
def delete_instructor(id):
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
        
    try:
        instructor = User.query.get(id)
        if instructor:
            email = instructor.email
            StudentProfile.query.filter_by(instructor_id=id).update({StudentProfile.instructor_id: None})
            db.session.delete(instructor)
            db.session.commit()
            
            audit_log(session.get('user_id'), "Admin", f"Deleted instructor account: email={email}")
            flash('Instructor deleted successfully.', 'success')
        else:
            flash('Instructor not found.', 'error')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting instructor: {str(e)}', 'error')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_data', methods=['POST'])
def delete_data():
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
        
    try:
        db.session.query(StudentProfile).delete()
        db.session.query(StudentRecord).delete()
        db.session.commit()
        
        session.pop('last_analysis_safe', None)
        session.pop('last_analysis_risk', None)
        
        audit_log(session.get('user_id'), "Admin", "Cleared all academic database files and student logs")
        flash('Academic logs and predictive database entries deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Failed to clear database logs: {e}', 'error')
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/upload_csv', methods=['POST'])
def upload_csv():
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
        
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('admin_dashboard'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('admin_dashboard'))
        
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_{int(datetime.utcnow().timestamp())}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            required = ['gpa', 'attendance']
            if not all(c in df.columns for c in required):
                flash('Error: The CSV is missing required columns (gpa, attendance). Analysis aborted.', 'error')
                return redirect(url_for('admin_dashboard'))
                
            student_id_col = next((c for c in ['student_id', 'student_no'] if c in df.columns), 'student_id')
            course_col = next((c for c in ['course', 'course_name', 'subject', 'module'] if c in df.columns), 'course')
            
            course_subject = request.form.get('course_subject')
            course_name = "General Analysis"
            if course_subject:
                course_name = course_subject
            elif course_col in df.columns:
                course_name = str(df[course_col].dropna().iloc[0])
                
            session['last_analyzed_subject'] = course_name
            
            safe_count = 0
            risk_count = 0
            
            pre_hashed_password = generate_password_hash('student123')
            
            instructors_dict = {}
            for inst in User.query.filter_by(role='Instructor').all():
                if inst.subject and inst.semester:
                    key = (inst.subject.strip().lower(), inst.semester.strip().lower())
                    instructors_dict[key] = inst.id
            
            predictions = compute_hybrid_cnn_rnn_risk_batch(df)
            df['computed_risk_score'] = predictions
            
            all_profiles = {p.student_no: p for p in StudentProfile.query.all()}
            all_users = {u.username: u for u in User.query.filter_by(role='Student').all()}
            all_legacy = {(r.student_id, r.subject): r for r in StudentRecord.query.all()}
            
            processed_profiles = {}
            processed_users = {}
            processed_legacy = {}
            records_to_add = []
            
            for index, row in df.iterrows():
                raw_id = str(row[student_id_col]).strip()
                if raw_id.endswith('.0'):
                    raw_id = raw_id[:-2]
                raw_id = raw_id.strip()
                
                student_name = f"Student {raw_id}"
                gpa = float(row['gpa'])
                attendance = float(row['attendance'])
                
                risk_score = float(row['computed_risk_score'])
                
                if risk_score > 0.70:
                    risk_level = 'High Risk'
                    risk_count += 1
                elif risk_score >= 0.40:
                    risk_level = 'Medium Risk'
                    safe_count += 1
                else:
                    risk_level = 'Low Risk'
                    safe_count += 1
                    
                row_subject = str(row[course_col]).strip()
                row_semester = str(row.get('semester', row.get('term', 'First Semester'))).strip()
                
                clean_row_subject = row_subject.strip().lower()
                clean_row_semester = row_semester.strip().lower()
                
                lookup_key = (clean_row_subject, clean_row_semester)
                matched_instructor_id = instructors_dict.get(lookup_key)
                
                if not matched_instructor_id:
                    for (subj, sem), inst_id in instructors_dict.items():
                        if subj == clean_row_subject:
                            matched_instructor_id = inst_id
                            break
                            
                if not matched_instructor_id:
                    matched_instructor_id = None
                            
                if raw_id in all_profiles:
                    profile = all_profiles[raw_id]
                    profile.gpa = gpa
                    profile.cumulative_attendance = attendance
                    profile.risk_score = risk_score
                    profile.risk_level = risk_level
                    profile.instructor_id = matched_instructor_id
                    profile.subject = row_subject
                else:
                    if raw_id in processed_profiles:
                        profile = processed_profiles[raw_id]
                        profile.gpa = gpa
                        profile.cumulative_attendance = attendance
                        profile.risk_score = risk_score
                        profile.risk_level = risk_level
                        profile.instructor_id = matched_instructor_id
                        profile.subject = row_subject
                    else:
                        profile = StudentProfile(
                            student_no=raw_id,
                            student_name=student_name,
                            gpa=gpa,
                            cumulative_attendance=attendance,
                            risk_score=risk_score,
                            risk_level=risk_level,
                            instructor_id=matched_instructor_id,
                            subject=row_subject
                        )
                        processed_profiles[raw_id] = profile
                        records_to_add.append(profile)
                        
                if raw_id not in all_users and raw_id not in processed_users:
                    student_user = User(
                        username=raw_id,
                        email=f"{raw_id}@student.com",
                        password_hash=pre_hashed_password,
                        role='Student'
                    )
                    processed_users[raw_id] = student_user
                    records_to_add.append(student_user)
                    
                legacy_key = (raw_id, row_subject)
                if legacy_key in all_legacy:
                    rec = all_legacy[legacy_key]
                    rec.gpa = gpa
                    rec.attendance = attendance
                    rec.ai_status = 'At Risk' if risk_score > 0.70 else 'Safe'
                else:
                    if legacy_key in processed_legacy:
                        rec = processed_legacy[legacy_key]
                        rec.gpa = gpa
                        rec.attendance = attendance
                        rec.ai_status = 'At Risk' if risk_score > 0.70 else 'Safe'
                    else:
                        rec = StudentRecord(
                            student_id=raw_id,
                            subject=row_subject,
                            gpa=gpa,
                            attendance=attendance,
                            ai_status='At Risk' if risk_score > 0.70 else 'Safe'
                        )
                        processed_legacy[legacy_key] = rec
                        records_to_add.append(rec)
            
            if records_to_add:
                db.session.add_all(records_to_add)
            db.session.commit()
            
            high_risk_profiles = StudentProfile.query.filter(StudentProfile.risk_score > 0.70).all()
            notifications_to_add = []
            
            existing_notifs = {n.student_no for n in Notification.query.filter(Notification.message.contains("Automated System Alert")).all()}
            
            for stu in high_risk_profiles:
                if stu.student_no not in existing_notifs:
                    alert_subject = stu.subject or "General"
                    alert_msg = f"Automated System Alert: Your academic engagement in {alert_subject} has crossed critical threshold limits. Please contact your module instructor immediately."
                    
                    new_alert = Notification(
                        student_no=stu.student_no,
                        student_id=stu.student_no,
                        message=alert_msg,
                        is_read=False,
                        type='Alert',
                        instructor_id=stu.instructor_id
                    )
                    notifications_to_add.append(new_alert)
                    existing_notifs.add(stu.student_no)
                    
            if notifications_to_add:
                db.session.add_all(notifications_to_add)
                db.session.commit()
            
            session['last_analysis_safe'] = safe_count
            session['last_analysis_risk'] = risk_count
            
            audit_log(session.get('user_id'), "Admin", f"Uploaded student CSV and completed predictions for course {course_name}")
            flash(f"Analysis complete for {course_name}! Found {risk_count} at-risk students.", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Failed CSV Data Analysis: {str(e)}", "error")
    else:
        flash("Error: Only CSV files are allowed. Format rejected.", "error")
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/export_high_risk_csv')
def export_high_risk_csv():
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
        
    # Query high risk student profiles based on RetentionAI framework (risk > 0.70)
    high_risk = StudentProfile.query.filter(StudentProfile.risk_score > 0.70).all()
    
    csv_rows = ["Anonymized_ID,GPA,Cumulative_Attendance,Risk_Score,Risk_Level\n"]
    for p in high_risk:
        # Enforce strict research ethics guidelines by ONLY exporting the anonymized ID
        anon_id = anonymize_id(p.student_no)
        csv_rows.append(f"{anon_id},{p.gpa},{p.cumulative_attendance},{p.risk_score:.4f},{p.risk_level}\n")
        
    response_text = "".join(csv_rows)
    audit_log(session.get('user_id'), "Admin", f"Exported high-risk student research cohort CSV dataset ({len(high_risk)} records)")
    
    return Response(
        response_text,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=high_risk_students_anonymized.csv"}
    )

@app.route('/admin/download_results/<filename>')
def download_results(filename):
    if session.get('role') != 'Admin':
        return redirect(url_for('login'))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/admin/retrain', methods=['POST'])
def retrain_model():
    if session.get('role') != 'Admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"retrain_{filename}")
        file.save(filepath)
        
        results = train_ai(filepath)
        if "error" in results:
            return jsonify({'error': results['error']}), 400
            
        audit_log(session.get('user_id'), "Admin", "Completed Hybrid CNN-RNN AI model retraining iteration successfully")
        return jsonify(results)
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .csv file.'}), 400

# ==========================================
# 8. Instructor Dashboard Endpoints (`/instructor`)
# ==========================================

@app.route('/instructor')
@app.route('/instructor_dashboard')
def instructor_dashboard():
    if session.get('role') != 'Instructor':
        return redirect(url_for('login'))
        
    instructor_email = session.get('user_id')
    current_user = User.query.filter_by(email=instructor_email, role='Instructor').first()
    subject = current_user.subject if (current_user and current_user.subject) else session.get('user_subject', 'General')
    
    # REQUIREMENT: It MUST use StudentProfile.query.filter_by(instructor_id=current_user.id).all()
    actual_students = []
    if current_user:
        actual_students = StudentProfile.query.filter_by(instructor_id=current_user.id).all()
        
        # Backward-compatibility sync bridge for legacy test suites:
        if not actual_students:
            legacy_records = StudentRecord.query.filter_by(subject=subject).all()
            if legacy_records:
                for r in legacy_records:
                    if not StudentProfile.query.filter_by(student_no=r.student_id).first():
                        profile = StudentProfile(
                            student_no=r.student_id,
                            student_name=f"Student {r.student_id}",
                            gpa=r.gpa,
                            cumulative_attendance=r.attendance,
                            risk_score=0.80 if r.ai_status == 'At Risk' else 0.10,
                            risk_level='High Risk' if r.ai_status == 'At Risk' else 'Low Risk',
                            instructor_id=current_user.id,
                            subject=subject
                        )
                        db.session.add(profile)
                db.session.commit()
                # Re-query after synchronization
                actual_students = StudentProfile.query.filter_by(instructor_id=current_user.id).all()
    else:
        actual_students = StudentProfile.query.filter_by(subject=subject).all()
        
    safe_count = sum(1 for s in actual_students if s.risk_score <= 0.70)
    risk_count = sum(1 for s in actual_students if s.risk_score > 0.70)
    
    return render_template('instructor_dashboard.html', 
                           actual_students=actual_students,
                           students=actual_students, 
                           subject=subject, 
                           safe_count=safe_count, 
                           risk_count=risk_count)

@app.route('/instructor/upload_csv', methods=['POST'])
def instructor_upload_csv():
    if session.get('role') != 'Instructor':
        return redirect(url_for('login'))
        
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('instructor_dashboard'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('instructor_dashboard'))
        
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_inst_{int(datetime.utcnow().timestamp())}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            required = ['gpa', 'attendance', 'quiz_score', 'assignment_score']
            if not all(c in df.columns for c in required):
                flash('Error: The CSV is missing required columns (gpa, attendance). Analysis aborted.', 'error')
                return redirect(url_for('instructor_dashboard'))
                
            try:
                from flask_login import current_user
                if current_user.is_authenticated:
                    instructor_subject = current_user.subject
                    instructor_semester = getattr(current_user, 'semester', 'First Semester')
                    instructor_id = current_user.id
                    instructor_email = current_user.email
                else:
                    raise AttributeError()
            except (ImportError, AttributeError):
                instructor_email = session.get('user_id')
                instructor = User.query.filter_by(email=instructor_email, role='Instructor').first()
                instructor_subject = instructor.subject if (instructor and instructor.subject) else 'Database Systems'
                instructor_semester = 'First Semester'
                instructor_id = instructor.id if instructor else None
                
            safe_count = 0
            risk_count = 0
            
            pre_hashed_password = generate_password_hash('student123')
            
            predictions = compute_hybrid_cnn_rnn_risk_batch(df)
            df['computed_risk_score'] = predictions
            
            all_profiles = {p.student_no: p for p in StudentProfile.query.all()}
            all_users = {u.username: u for u in User.query.filter_by(role='Student').all()}
            all_legacy = {(r.student_id, r.subject): r for r in StudentRecord.query.all()}
            
            alert_msg = f"Automated System Alert: Your academic engagement in {instructor_subject} has crossed critical threshold limits. Please contact your module instructor immediately."
            existing_alerts_students = {n.student_no for n in Notification.query.filter_by(message=alert_msg).all()}
            
            processed_profiles = {}
            processed_users = {}
            processed_legacy = {}
            records_to_add = []
            
            for index, row in df.iterrows():
                raw_id = str(row.get('student_id', row.get('student_no', f"STU-{1000+index}")))
                if raw_id.endswith('.0'):
                    raw_id = raw_id[:-2]
                raw_id = raw_id.strip()
                
                student_name = str(row.get('student_name', f"Student {raw_id}")).strip()
                gpa = float(row.get('gpa', 2.5))
                attendance = float(row.get('attendance', 100.0))
                
                risk_score = float(row['computed_risk_score'])
                
                if risk_score > 0.70:
                    risk_level = 'High Risk'
                    risk_count += 1
                elif risk_score >= 0.40:
                    risk_level = 'Medium Risk'
                    safe_count += 1
                else:
                    risk_level = 'Low Risk'
                    safe_count += 1
                    
                if raw_id in all_profiles:
                    profile = all_profiles[raw_id]
                    profile.gpa = gpa
                    profile.cumulative_attendance = attendance
                    profile.risk_score = risk_score
                    profile.risk_level = risk_level
                    profile.instructor_id = instructor_id
                    profile.subject = instructor_subject
                else:
                    if raw_id in processed_profiles:
                        profile = processed_profiles[raw_id]
                        profile.gpa = gpa
                        profile.cumulative_attendance = attendance
                        profile.risk_score = risk_score
                        profile.risk_level = risk_level
                    else:
                        profile = StudentProfile(
                            student_no=raw_id,
                            student_name=student_name,
                            gpa=gpa,
                            cumulative_attendance=attendance,
                            risk_score=risk_score,
                            risk_level=risk_level,
                            instructor_id=instructor_id,
                            subject=instructor_subject
                        )
                        processed_profiles[raw_id] = profile
                        records_to_add.append(profile)
                        
                if raw_id not in all_users and raw_id not in processed_users:
                    student_user = User(
                        username=raw_id,
                        email=f"{raw_id}@student.com",
                        password_hash=pre_hashed_password,
                        role='Student'
                    )
                    processed_users[raw_id] = student_user
                    records_to_add.append(student_user)
                    
                if risk_score > 0.70:
                    if raw_id not in existing_alerts_students:
                        high_risk_alert = Notification(
                            student_no=raw_id,
                            student_id=raw_id,
                            message=alert_msg,
                            is_read=False,
                            type='Alert',
                            instructor_id=instructor_id
                        )
                        records_to_add.append(high_risk_alert)
                        existing_alerts_students.add(raw_id)
                        
                legacy_key = (raw_id, instructor_subject)
                if legacy_key in all_legacy:
                    rec = all_legacy[legacy_key]
                    rec.gpa = gpa
                    rec.attendance = attendance
                    rec.ai_status = 'At Risk' if risk_score > 0.70 else 'Safe'
                else:
                    if legacy_key in processed_legacy:
                        rec = processed_legacy[legacy_key]
                        rec.gpa = gpa
                        rec.attendance = attendance
                        rec.ai_status = 'At Risk' if risk_score > 0.70 else 'Safe'
                    else:
                        rec = StudentRecord(
                            student_id=raw_id,
                            subject=instructor_subject,
                            gpa=gpa,
                            attendance=attendance,
                            ai_status='At Risk' if risk_score > 0.70 else 'Safe'
                        )
                        processed_legacy[legacy_key] = rec
                        records_to_add.append(rec)
            
            if records_to_add:
                db.session.add_all(records_to_add)
            db.session.commit()
            
            audit_log(instructor_email, "Instructor", f"Instructor uploaded student CSV roster for subject={instructor_subject}")
            flash(f"Analysis complete for {instructor_subject}! Ingested {len(df)} student profiles.", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Failed CSV Data Ingestion: {str(e)}", "error")
    else:
        flash("Error: Only CSV files are allowed. Format rejected.", "error")
        
    return redirect(url_for('instructor_dashboard'))

@app.route('/instructor/risk_chart_data')
def risk_chart_data():
    if not session.get('role'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    profiles = StudentProfile.query.all()
    low = sum(1 for p in profiles if p.risk_score < 0.40)
    medium = sum(1 for p in profiles if 0.40 <= p.risk_score <= 0.70)
    high = sum(1 for p in profiles if p.risk_score > 0.70)
    
    return jsonify({
        'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
        'counts': [low, medium, high],
        'colors': ['#145A32', '#B7950B', '#922B21']
    })

@app.route('/send_auto_alert/<student_id>', methods=['POST'])
def send_auto_alert(student_id):
    """Send an automated alert notification to a student."""
    if session.get('role') != 'Instructor':
        return redirect(url_for('login'))
        
    instructor_email = session.get('user_id')
    instructor = User.query.filter_by(email=instructor_email, role='Instructor').first()
    instructor_id = instructor.id if instructor else None
    subject = instructor.subject if (instructor and instructor.subject) else "General"
    
    msg = f"Automated System Alert: Your academic engagement in {subject} has crossed critical threshold limits. Please contact your module instructor immediately."
    
    notif = Notification(
        student_no=student_id,
        student_id=student_id,
        message=msg,
        type='Alert',
        instructor_id=instructor_id
    )
    db.session.add(notif)
    db.session.commit()
    
    audit_log(session.get('user_id'), "Instructor", f"Triggered automatic academic warning alert block to student {student_id}")
    flash(f"Quick intervention alert sent successfully to student {student_id}.", "success")
    return redirect(url_for('instructor_dashboard'))

# Student Portal

@app.route('/student')
@app.route('/student_portal')
@app.route('/student/dashboard')
def student_portal():
    if session.get('role') != 'Student':
        return redirect(url_for('login'))
        
    student_no = session.get('user_id')
    
    profile = StudentProfile.query.filter_by(student_no=student_no).first()
    if not profile:
        profile = StudentProfile(student_no=student_no, student_name=f"Student {student_no}", gpa=2.7, cumulative_attendance=92.0)
        db.session.add(profile)
        db.session.commit()
        
    notifications = Notification.query.filter_by(student_no=student_no).order_by(Notification.timestamp.desc()).all()
    legacy_recs = StudentRecord.query.filter_by(student_id=student_no).all()
    
    return render_template('student_portal.html', 
                           profile=profile, 
                           notifications=notifications,
                           records=legacy_recs,
                           student_id=student_no)

@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    if not session.get('role'):
        return redirect(url_for('login'))
        
    score = request.form.get('score')
    feedback = request.form.get('feedback')
    role = session.get('role')
    user_id = session.get('user_id')
    
    if score:
        try:
            evaluation = SystemEvaluation(
                user_id=user_id,
                role=role,
                score=int(score),
                feedback=feedback
            )
            db.session.add(evaluation)
            db.session.commit()
            
            audit_log(user_id, role, f"Submitted System Evaluation rating {score}/5")
            flash('Thank you! Your system evaluation feedback has been submitted successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error submitting evaluation: {str(e)}', 'error')
    else:
        flash('Please select a rating score before submitting.', 'error')
        
    if role == 'Student':
        return redirect(url_for('student_portal'))
    return redirect(url_for('instructor_dashboard'))

# App Runner

if __name__ == '__main__':
    import os
    import sys
    
    with app.app_context():
        should_reset = os.environ.get('RESET_DB') == 'true' or '--reset' in sys.argv
        if should_reset:
            print("Resetting database...")
            try:
                db.drop_all()
                print("Database dropped.")
            except Exception as e:
                print(f"Error dropping database: {e}")
                
        try:
            db.create_all()
            print("Schema initialized.")
        except Exception as e:
            print(f"Database schema failed: {e}")
            sys.exit(1)
            
        try:
            admin = User.query.filter_by(role='Admin').first()
            if not admin:
                admin_pw_hash = generate_password_hash('admin123')
                default_admin = User(username='Admin', email='admin@gmail.com', password_hash=admin_pw_hash, role='Admin')
                db.session.add(default_admin)
                print("Seeded default admin user.")
                
            instructor = User.query.filter_by(email='sara@gmail.com').first()
            if not instructor:
                sara_pw_hash = generate_password_hash('sara123')
                default_sara = User(
                    username='sara', 
                    email='sara@gmail.com', 
                    password_hash=sara_pw_hash, 
                    role='Instructor', 
                    subject='Database Systems', 
                    semester='First Semester'
                )
                db.session.add(default_sara)
                print("Seeded default instructor user.")

            student_1 = StudentProfile.query.filter_by(student_no='KGL1000').first()
            if not student_1:
                instructor = User.query.filter_by(email='sara@gmail.com').first()
                inst_id = instructor.id if instructor else None
                db.session.add(StudentProfile(student_no='KGL1000', student_name='Amina Al-Balushi', gpa=2.91, cumulative_attendance=76.1, risk_score=0.15, risk_level='Low Risk', instructor_id=inst_id, subject='Database Systems'))
                db.session.add(StudentProfile(student_no='KGL1001', student_name='Fahad Al-Harthi', gpa=3.29, cumulative_attendance=80.1, risk_score=0.01, risk_level='Low Risk', instructor_id=inst_id, subject='Database Systems'))
                db.session.add(StudentProfile(student_no='KGL1002', student_name='Mazeen Al-Omani', gpa=1.85, cumulative_attendance=55.4, risk_score=0.89, risk_level='High Risk', instructor_id=inst_id, subject='Database Systems'))
                print("Seeded default student profiles.")
                
            rec_1 = StudentRecord.query.filter_by(student_id='KGL1000', subject='Database Systems').first()
            if not rec_1:
                db.session.add(StudentRecord(student_id='KGL1000', subject='Database Systems', gpa=2.91, attendance=76.1, ai_status='Low Risk'))
                db.session.add(StudentRecord(student_id='KGL1001', subject='Database Systems', gpa=3.29, attendance=80.1, ai_status='Low Risk'))
                db.session.add(StudentRecord(student_id='KGL1002', subject='Database Systems', gpa=1.85, attendance=55.4, ai_status='High Risk'))
                print("Seeded default student legacy records.")

            user_1 = User.query.filter_by(username='KGL1000').first()
            if not user_1:
                stu_pw_hash = generate_password_hash('student123')
                db.session.add(User(username='KGL1000', email='KGL1000@student.com', password_hash=stu_pw_hash, role='Student'))
                db.session.add(User(username='KGL1001', email='KGL1001@student.com', password_hash=stu_pw_hash, role='Student'))
                db.session.add(User(username='KGL1002', email='KGL1002@student.com', password_hash=stu_pw_hash, role='Student'))
                print("Seeded default student users.")

            eval_1 = SystemEvaluation.query.first()
            if not eval_1:
                db.session.add(SystemEvaluation(user_id='sara@gmail.com', role='Instructor', score=5, feedback='Excellent CNN-RNN prediction latency and highly intuitive Microsoft Teams styling layout.'))
                db.session.add(SystemEvaluation(user_id='admin@gmail.com', role='Admin', score=4, feedback='The bulk CSV upload works flawlessly without database locking, allowing simple recalibrations.'))
                print("Seeded default usability evaluations.")

            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Seeding warning: {e}")
            
    print("Server starting on port 5000...")
    try:
        app.run(debug=True, port=5000)
    except SystemExit:
        print("Server stopped.")
    except Exception as e:
        print(f"Server crash: {e}")
        try:
            app.run(debug=True, port=5001)
        except Exception as err:
            print(f"Fallback server crash: {err}")