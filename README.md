# 🎓 Student Academic Prediction & Retention System (RetentionAI)

An integrated intelligent system designed for the early prediction of academic dropout or student failure using a hybrid artificial intelligence model (**Hybrid CNN-RNN**). The system aims to enable administrators and educators to take prompt preventive actions to support students and improve academic retention rates.

---

## 🌟 Key Features

### 🧑‍💼 System Administrator Dashboard
*   **Real-time Data Analysis:** Upload student datasets in `.csv` format to instantly classify students into risk levels (High, Medium, Low) using the machine learning model.
*   **Analytics & Visualizations:** Access interactive dashboards showing dropout risk distributions and statistical breakdown per academic subject.
*   **Instructor Management:** Register, view, and delete instructors, and assign them to specific subjects and semesters.
*   **Model Performance Metrics:** View deep learning model metrics (Accuracy, Precision, Recall, F1-Score) alongside an interactive Confusion Matrix.
*   **System Evaluation:** Display usability feedback submitted by students and instructors, calculating a weighted average score.
*   **System Audit Logs:** A comprehensive, non-editable security audit trail tracking all actions and events performed within the system.

### 👨‍🏫 Instructor Portal
*   **Student Monitoring:** Monitor students enrolled specifically in the instructor's assigned course (including attendance rates, GPA, and AI-predicted risk levels).
*   **Quick Intervention:** Send immediate academic warning alerts directly to a student's portal with a single click if they are flagged as "At Risk".
*   **System Evaluation:** Submit feedback, ratings, and comments regarding the system's performance to the administration.

### 🎓 Student Portal
*   **Academic Card:** View personal academic profiles, performance details, and grades.
*   **Notifications Box:** Receive guidance messages and academic warning alerts sent by module instructors.
*   **Feedback & Usability:** Submit experience ratings and notes directly to the evaluation dashboard.

---

## 🛠️ Technology Stack

*   **Backend Framework:** Flask (Python)
*   **Database:** SQLite (default for local development) / supports MySQL database instances.
*   **Machine Learning & DL:** TensorFlow, Keras (Hybrid CNN-RNN model), Scikit-learn
*   **Data Processing:** Pandas, NumPy
*   **UI/UX Design:** HTML5, CSS3 (Modern Glassmorphism UI styling with responsive layouts)
*   **Testing Suite:** Pytest

---

## 🚀 Local Installation & Setup

### Prerequisites
*   Python 3.8 or higher.

### Steps to Run

1.  **Install Dependencies:**
    Open your command prompt or terminal in the project's root directory and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Reset and Seed the Database:**
    To initialize database tables and seed them with default demo accounts, run:
    ```bash
    python app.py --reset
    ```

3.  **Start the Local Server:**
    Start the Flask development server:
    ```bash
    python app.py
    ```
    The application will run locally at: `http://127.0.0.1:5000/`

---

## 🔐 Approved Test Credentials

The system comes pre-configured with the following credentials to simplify the testing process:

| Role | Username / Email | Password |
| :--- | :--- | :--- |
| **Admin** | `admin@gmail.com` | `admin123` |
| **Instructor** | `instructor@test.com` | `instructor123` |
| **Student** | `KGL1001` (Student ID) | `student123` |

---

## 🧪 Automated Testing System

To verify code quality and database consistency, run the suite of 26 automated unit tests covering login verification, file upload validation, and role access controls:
```bash
pytest test_app.py -v
```
