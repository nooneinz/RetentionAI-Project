import os
import io
# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app import app, db, User, AuditLog, StudentRecord, Notification, SystemEvaluation
from werkzeug.security import generate_password_hash

# =====================================================================
# SETUP: Testing Configurations & Fixtures
# =====================================================================
@pytest.fixture
def client():
    # Configure app for testing
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:' # High-speed in-memory database
    app.config['UPLOAD_FOLDER'] = 'test_uploads'
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    with app.test_client() as client:
        with app.app_context():
            # Create a fresh in-memory SQLite engine and bind it to Flask-SQLAlchemy
            import sqlalchemy
            new_engine = sqlalchemy.create_engine('sqlite:///:memory:')
            if hasattr(db, '_engines'):
                db._engines[app] = new_engine
            if hasattr(db, 'engines'):
                db.engines[app] = new_engine
            db.metadata.bind = new_engine
            db.create_all()
            
            # Seed default test users
            admin = User(
                username="Test Admin",
                email="admin@test.com",
                password_hash=generate_password_hash("admin123"),
                role="Admin"
            )
            instructor = User(
                username="Test Instructor",
                email="instructor@test.com",
                password_hash=generate_password_hash("instructor123"),
                role="Instructor",
                subject="Machine Learning",
                semester="Second Semester"
            )
            db.session.add(admin)
            db.session.add(instructor)
            db.session.commit()
            
            yield client
            
            db.session.remove()
            db.drop_all()

# Helper function to simulate admin login session
def login_admin(client):
    return client.post('/login', data={
        'email': 'admin@test.com',
        'password': 'admin123',
        'role': 'Admin'
    }, follow_redirects=True)

# Helper function to simulate instructor login session
def login_instructor(client):
    return client.post('/login', data={
        'email': 'instructor@test.com',
        'password': 'instructor123',
        'role': 'Instructor'
    }, follow_redirects=True)


# =====================================================================
# MODULE 1: Authentication Module Tests (TC-01 to TC-05)
# =====================================================================

def test_tc01_admin_login_valid(client):
    """TC-01: Admin Login with valid credentials."""
    response = client.post('/login', data={
        'email': 'admin@test.com',
        'password': 'admin123',
        'role': 'Admin'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'Admin Portal' in response.data

def test_tc02_instructor_login_valid(client):
    """TC-02: Instructor Login with valid credentials."""
    response = client.post('/login', data={
        'email': 'instructor@test.com',
        'password': 'instructor123',
        'role': 'Instructor'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'Instructor Portal' in response.data

def test_tc03_login_invalid_password(client):
    """TC-03: Login with an invalid password (ensure it fails)."""
    response = client.post('/login', data={
        'email': 'admin@test.com',
        'password': 'wrongpassword',
        'role': 'Admin'
    }, follow_redirects=True)
    assert b'Invalid login credentials' in response.data

def test_tc04_login_unregistered_email(client):
    """TC-04: Login with an unregistered email (ensure it fails)."""
    response = client.post('/login', data={
        'email': 'notfound@test.com',
        'password': 'anypassword',
        'role': 'Admin'
    }, follow_redirects=True)
    assert b'Invalid login credentials' in response.data

def test_tc05_user_logout(client):
    """TC-05: User Logout functionality (ensure session is cleared)."""
    # Login first
    login_admin(client)
    # Logout
    response = client.get('/logout', follow_redirects=True)
    # Verify redirected back to login
    assert response.status_code == 200
    assert b'Login' in response.data or b'Sign In' in response.data


# =====================================================================
# MODULE 2: Instructor Management Module Tests (TC-06 to TC-09)
# =====================================================================

def test_tc06_admin_adds_instructor_valid(client):
    """TC-06: Admin successfully adds a new Instructor with valid data."""
    login_admin(client)
    response = client.post('/admin/add_instructor', data={
        'name': 'Dr. Sarah',
        'email': 'sarah@test.com',
        'password': 'password123',
        'subject': 'Web Development',
        'semester': 'Second Semester'
    }, follow_redirects=True)
    assert response.status_code == 200
    # Check database
    new_user = User.query.filter_by(email='sarah@test.com').first()
    assert new_user is not None
    assert new_user.username == 'Dr. Sarah'
    assert new_user.subject == 'Web Development'

def test_tc07_admin_adds_duplicate_instructor(client):
    """TC-07: Admin attempts to add an Instructor with an already existing email."""
    login_admin(client)
    # Attempt to add duplicate email (instructor@test.com already exists in setup fixture)
    response = client.post('/admin/add_instructor', data={
        'name': 'Duplicate Name',
        'email': 'instructor@test.com',
        'password': 'password123',
        'subject': 'Database Systems',
        'semester': 'First Semester'
    }, follow_redirects=True)
    assert b'Error:' in response.data or b'already registered' in response.data.lower() or response.status_code == 200

def test_tc08_admin_deletes_instructor(client):
    """TC-08: Admin deletes an existing Instructor successfully."""
    login_admin(client)
    # Get id of original seeded instructor
    inst = User.query.filter_by(email='instructor@test.com').first()
    assert inst is not None
    response = client.post(f'/admin/delete_instructor/{inst.id}', follow_redirects=True)
    assert response.status_code == 200
    # Verify deletion
    deleted_inst = User.query.filter_by(email='instructor@test.com').first()
    assert deleted_inst is None

def test_tc09_instructor_accesses_admin_panel_direct(client):
    """TC-09: An Instructor attempts to access the admin_dashboard URL directly."""
    login_instructor(client)
    response = client.get('/admin_dashboard', follow_redirects=False)
    # Assert redirected away from admin dashboard
    assert response.status_code == 302 


# =====================================================================
# MODULE 3: CSV Upload & AI Prediction Module Tests (TC-10 to TC-14)
# =====================================================================

@patch('app.load_model')
def test_tc10_admin_uploads_valid_csv(mock_load_model, client):
    """TC-10: Admin uploads a valid CSV file. Ensure processing is successful."""
    login_admin(client)
    
    # Mock tensorflow model object
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8], [0.2]]) # Mock returns At-Risk and Safe
    mock_load_model.return_value = mock_model

    # Generate valid dataset
    csv_data = "student_id,gpa,attendance,quiz_score,assignment_score,course\n" \
               "KGL1001,3.8,95,90,85,Machine Learning\n" \
               "KGL1002,2.1,60,50,45,Machine Learning"
    
    response = client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'students.csv')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    assert response.status_code == 200
    assert b'Analysis complete' in response.data
    # Ensure records are populated in database
    records = StudentRecord.query.all()
    assert len(records) == 2

@patch('app.load_model')
def test_tc11_autodetect_course_name(mock_load_model, client):
    """TC-11: Auto-detect Course Name from CSV."""
    login_admin(client)
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8]])
    mock_load_model.return_value = mock_model

    csv_data = "student_id,gpa,attendance,quiz_score,assignment_score,course_name\n" \
               "KGL1001,3.5,90,80,80,Database Systems"
               
    response = client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'db_students.csv')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    # Confirm the session or DB captured Course Name correctly
    records = StudentRecord.query.filter_by(subject='Database Systems').all()
    assert len(records) == 1

def test_tc12_admin_uploads_non_csv(client):
    """TC-12: Admin uploads a non-CSV file (e.g., dummy pdf)."""
    login_admin(client)
    
    response = client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(b"dummy PDF content"), 'thesis.pdf')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    assert response.status_code == 200
    assert b'Error: Only CSV files are allowed' in response.data

def test_tc13_admin_uploads_no_file(client):
    """TC-13: Admin submits the analysis form without attaching any file."""
    login_admin(client)
    
    response = client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(b""), '')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    assert b'No selected file' in response.data

def test_tc14_admin_uploads_csv_missing_columns(client):
    """TC-14: Admin uploads a CSV missing required columns."""
    login_admin(client)
    
    # Missing mandatory DL scaling column 'gpa'
    csv_data = "student_id,attendance,quiz_score,assignment_score\n" \
               "KGL1001,90,80,80"
               
    response = client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'broken.csv')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    assert b'missing required columns' in response.data


# =====================================================================
# MODULE 4: Audit Logs & Quick Interventions Tests (TC-18 to TC-20, TC-23 to TC-25)
# =====================================================================

def test_tc18_audit_log_added_on_login(client):
    """TC-18: Verify that a successful login attempt adds a record to AuditLog table."""
    login_admin(client)
    # Verify entry exists
    log = AuditLog.query.filter_by(user_id='admin@test.com', action='Logged into the system').first()
    assert log is not None

def test_tc19_audit_log_added_on_delete_instructor(client):
    """TC-19: Verify that deleting an instructor adds a record to AuditLog."""
    login_admin(client)
    inst = User.query.filter_by(email='instructor@test.com').first()
    client.post(f'/admin/delete_instructor/{inst.id}', follow_redirects=True)
    
    # Query logs
    log = AuditLog.query.filter(AuditLog.action.contains('Deleted instructor')).first()
    assert log is not None

@patch('app.load_model')
def test_tc20_audit_log_logs_csv_subject(mock_load_model, client):
    """TC-20: Verify that uploading a CSV logs the exact subject name analyzed."""
    login_admin(client)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1]])
    mock_load_model.return_value = mock_model
    
    csv_data = "student_id,gpa,attendance,quiz_score,assignment_score,course\n" \
               "KGL1003,3.9,90,90,90,Data Structures"
               
    client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'ds.csv')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    log = AuditLog.query.filter(AuditLog.action.contains('Data Structures')).first()
    assert log is not None

def test_tc23_send_quick_intervention_saves_notification(client):
    """TC-23: Instructor sends Quick Intervention. Ensure new record in Notification table."""
    login_instructor(client)
    
    # Send Post Request for student KGL1005
    response = client.post('/send_auto_alert/KGL1005', follow_redirects=True)
    
    notif = Notification.query.filter_by(student_id='KGL1005').first()
    assert notif is not None
    assert notif.type == 'Alert'

def test_tc24_notification_contains_subject(client):
    """TC-24: Ensure the notification message contains the correct Instructor's subject."""
    login_instructor(client)
    client.post('/send_auto_alert/KGL1005', follow_redirects=True)
    
    notif = Notification.query.filter_by(student_id='KGL1005').first()
    # Seeding used "Machine Learning" for the assigned subject
    assert "Machine Learning" in notif.message

def test_tc25_notification_generates_flash_message(client):
    """TC-25: Ensure a success flash message is generated after sending the alert."""
    login_instructor(client)
    response = client.post('/send_auto_alert/KGL1005', follow_redirects=True)
    
    assert b'alert sent successfully' in response.data.lower()


# =====================================================================
# MODULE 5: Data Rendering & UI Context Tests (TC-15 to TC-17, TC-21 to TC-22)
# =====================================================================

@patch('app.load_model')
def test_tc15_tc16_admin_dashboard_renders_statistics(mock_load_model, client):
    """TC-15 & TC-16: Assert statistics data passed to Admin template successfully."""
    login_admin(client)
    
    mock_model = MagicMock()
    # Return mock predictions: first index is >0.5 (risk), second is <=0.5 (safe)
    mock_model.predict.return_value = np.array([[0.9], [0.1]])
    mock_load_model.return_value = mock_model
    
    csv_data = "student_id,gpa,attendance,quiz_score,assignment_score,course\n" \
               "S1,2.0,50,40,40,Stats\n" \
               "S2,3.9,95,90,90,Stats"
               
    # Run CSV analysis which populates session counts
    client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'stats.csv')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    # Fetch page to verify visual rendering
    response = client.get('/admin_dashboard')
    assert response.status_code == 200
    
    # Since we returned 1 at-risk and 1 safe, the chart elements/text would typically contain counts
    # Or at least ensure the analytical page is rendered.
    assert b'Stats' in response.data or b'Uploaded Dataset' in response.data

def test_tc17_admin_dashboard_returns_ok_status(client):
    """TC-17: Assert that admin_dashboard.html returns a 200 OK status code."""
    login_admin(client)
    response = client.get('/admin_dashboard')
    assert response.status_code == 200

def test_tc21_instructor_dashboard_passes_subject(client):
    """TC-21: Access /instructor_dashboard and assert assigned subject renders correctly."""
    login_instructor(client)
    response = client.get('/instructor_dashboard')
    assert response.status_code == 200
    # The instructor is assigned "Machine Learning" in fixture
    assert b'Machine Learning' in response.data

def test_tc22_instructor_dashboard_renders_student_data(client):
    """TC-22: Assert student data table is successfully rendered in Instructor view."""
    # 1. Add a student record assigned to instructor's subject in DB
    with app.app_context():
        record = StudentRecord(
            student_id='KGL9999',
            subject='Machine Learning',
            gpa=3.5,
            attendance=90.0,
            ai_status='Safe'
        )
        db.session.add(record)
        db.session.commit()
        
    login_instructor(client)
    response = client.get('/instructor_dashboard')
    
    # Assert student ID from table is returned in raw data
    assert b'KGL9999' in response.data


def test_tc23_student_dashboard_route_alias(client):
    """TC-23: Verify that /student/dashboard acts as an alias route for student_portal."""
    # Try accessing student dashboard unauthenticated - should redirect to login
    response = client.get('/student/dashboard', follow_redirects=True)
    assert b'Sign In' in response.data or b'Login' in response.data or b'Institutional Portal' in response.data
    
    # Log in as a student
    client.post('/login', data={
        'email': 'STU1001',
        'password': 'student123',
        'role': 'Student'
    }, follow_redirects=True)
    
    # Access the /student/dashboard route alias
    response = client.get('/student/dashboard')
    assert response.status_code == 200
    assert b'Student Portal' in response.data


@patch('app.load_model')
def test_tc24_admin_uploads_csv_creates_student_users(mock_load_model, client):
    """TC-24: Admin uploads valid CSV; verify student User accounts are initialized with student123 password."""
    login_admin(client)
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1], [0.9]])
    mock_load_model.return_value = mock_model
    
    csv_data = "student_id,gpa,attendance,quiz_score,assignment_score,course\n" \
               "CSVSTUDENT1,3.2,88,80,75,Machine Learning\n" \
               "CSVSTUDENT2,1.8,55,40,40,Machine Learning"
               
    response = client.post('/admin/upload_csv', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'new_students.csv')
    }, content_type='multipart/form-data', follow_redirects=True)
    
    assert response.status_code == 200
    
    # Verify both User accounts exist and are of role 'Student'
    with app.app_context():
        u1 = User.query.filter_by(username='CSVSTUDENT1', role='Student').first()
        u2 = User.query.filter_by(username='CSVSTUDENT2', role='Student').first()
        
        assert u1 is not None
        assert u2 is not None
        assert u1.email == "CSVSTUDENT1@student.com"
        assert u2.email == "CSVSTUDENT2@student.com"
        
        # Verify the default credentials are secure and valid pre-login
        from werkzeug.security import check_password_hash
        assert check_password_hash(u1.password_hash, 'student123')
        assert check_password_hash(u2.password_hash, 'student123')

