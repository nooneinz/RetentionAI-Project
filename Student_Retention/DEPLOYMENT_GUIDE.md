# RetentionAI - Deployment and Handoff Guide

Welcome to the **RetentionAI** Student Retention & Predictive Analytics Portal. This document provides step-by-step instructions on how to run, reset, and deploy the system online.

---

## 1. Default Verification & Test Credentials

The database comes pre-seeded with clean test accounts to demonstrate all core user portals:

*   **System Administrator (Admin Portal)**:
    *   **Email:** `admin@gmail.com`
    *   **Password:** `admin123`
*   **Module Instructor (Lecturer Analytics)**:
    *   **Email:** `sara@gmail.com`
    *   **Password:** `sara123`
*   **Student (Personal Risk Profile)**:
    *   **Email:** `KGL1000@student.com`
    *   **Password:** `student123`

---

## 2. Local Execution and Setup

### Prerequisites
- Python 3.10 or higher.
- Installed virtual environment (recommended).

### Installation Steps

1.  **Extract / Navigate to Workspace**:
    Make sure you are in the root directory containing `app.py`.

2.  **Install Required Dependencies**:
    Run the following command to install all framework packages, machine learning tools, and database drivers:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Reset / Re-initialize Database (Pristine Slate)**:
    To clear out old transaction logs, uploaded coursework records, and reset the SQLite database to the default seeded profiles:
    ```bash
    python app.py --reset
    ```
    *Note: This command drops all tables, recreates the schema, and inserts the default clean Admin, Instructor, and Student records.*

4.  **Run Development Server**:
    Start the local Flask server:
    ```bash
    python app.py
    ```
    The application will bind to `http://127.0.0.1:5000` (or `http://127.0.0.1:5001` if port 5000 is occupied).

---

## 3. Database Configurations & Fallback Logic

RetentionAI supports dynamic database environments:
- **MySQL / Production Database**:
  By default, the application tries to locate a local MySQL server (`mysql+pymysql://root:@localhost/student_retention_db`).
- **Offline SQLite Fallback**:
  If a MySQL database connection is not detected or fails to connect, the system automatically falls back to a local SQLite database (`instance/final_system_v4.db`) to guarantee offline capabilities.

---

## 4. Online Cloud Deployment (Production)

The application has been prepared for hosting using standard WSGI containers like **Gunicorn**.

### Configuration Files
- **`Procfile`**: Specifies the startup instruction for production web workers:
  ```yaml
  web: gunicorn app:app
  ```
- **`requirements.txt`**: Includes production-ready dependencies (`gunicorn`, `pymysql`, and `cryptography` for secure MySQL handshakes).

### Step-by-Step Deployment (e.g., Render / Heroku)

1.  **Push Repository to GitHub**:
    Ensure all files (including the `Procfile` and `requirements.txt`) are committed to a GitHub repository.

2.  **Create a New Web Service**:
    - Log in to your cloud hosting console (e.g., **Render** at `render.com`).
    - Choose **New > Web Service** and connect your GitHub repository.

3.  **Configure Build & Start Settings**:
    - **Environment / Runtime**: `Python`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `gunicorn app:app`

4.  **Set Environment Variables (Optional)**:
    In the environment settings, configure:
    - `SECRET_KEY`: A secure random key (e.g., `graduation_project_2026`).
    - `DATABASE_URL` *(Only if connecting to a remote MySQL/Postgres database)*: Input the remote database connection URI. If left blank, the application will automatically initialize the local SQLite database.

5.  **Trigger Deploy**:
    The platform will install dependencies and start the WSGI server. Once finished, a live public URL will be provided.
