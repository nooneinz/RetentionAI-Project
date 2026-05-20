# 🎓 Student Retention System Operation and User Guide

Welcome to the comprehensive guide for running and testing the system based on the Hybrid Artificial Intelligence **(Hybrid CNN-RNN)** for early academic prediction and student retention. This document is designed to provide a detailed and visual walkthrough for supervisors and testers to run and examine all system features without the need for direct communication.

---

## 📂 1. Prerequisites & Environment

The system is built on Python and the **Flask** environment with a modern interactive glassmorphism UI.
* **Python Version:** Python 3.8 and above.
* **Hosting Platform:** The system supports local execution (Localhost) and is ready for immediate cloud deployment (Render / Heroku) thanks to the included `Procfile` and `requirements.txt` files.

---

## 🚀 2. How to Run (First-Time Setup)

To run the system on your local machine, please follow these simple steps:

1.  **Open Command Prompt:** Open your command prompt (CMD / PowerShell) and navigate to the project folder.
2.  **Install Libraries (One-time only):** Run the following command to install all required libraries (Flask, Pandas, TensorFlow, SQLAlchemy):
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Local Server:** Type the following command to start the system:
    ```bash
    python app.py
    ```
4.  **Open the System in Browser:** Open your web browser (Google Chrome is preferred) and navigate to the following URL:
    `http://127.0.0.1:5000/`

---

## 🔐 3. Default Login Credentials for Testing

The database has been pre-configured with the following secured accounts to simplify the immediate testing process:

| User Role | Email | Password |
| :--- | :--- | :--- |
| **Admin** | `admin@gmail.com` | `admin123` |
| **Instructor** | `instructor@test.com` | `instructor123` |
| **Student** | `KGL1001` (Use Student ID) | (Direct login with ID only) |

---

## 🛡️ 4. Detailed Walkthrough by Role

This section is designed to guide the supervisor through testing the three main roles in a logical workflow order:

### 🧑‍💼 First: Admin Dashboard
When logging in as an Admin, you will find an integrated interface containing the following tabs:

1.  **📁 Upload Data:**
    * Select a student data file in `.csv` format and click "Upload".
    * **AI Technology:** The system will automatically pass the data to the **CNN-RNN** model and instantly classify the students as (At-Risk) or (Safe).
    * Interactive charts will immediately appear showing the dropout ratios for each analyzed subject.
    * You can download the analyzed results as a CSV via the download button, or delete them to clear the data via the "Delete Data" button.
2.  **👨‍🏫 Manage Instructors:**
    * You can add a new instructor by entering their name and email, and assigning them to a specific subject and semester from the academic dropdown menu.
    * You can delete registered instructors and monitor them in a premium-styled table.
3.  **🤖 Model Metrics:**
    * *Specific to Chapter 5:* Displays digital KPI cards for the hybrid model showing precise accuracy (Accuracy, Precision, Recall, F1-Score).
    * Displays an **Interactive Confusion Matrix** with academic color-coding to explain the mathematical classification performance (TP, TN, FP, FN).
4.  **📝 System Evaluation:**
    * *Specific to Chapter 5 for academic research evaluation:* Displays the **Weighted Mean** of the aggregated 5-point scale evaluations, its academic interpretation, evaluator type (students/instructors), and their comments in a dedicated table.
5.  **⚙️ System Logs:**
    * Displays a live, non-editable audit trail of every action taken within the system for cybersecurity and activity tracking purposes.

---

### 👨‍🏫 Second: Instructor Portal
When logging in with a registered instructor account (e.g., `instructor@test.com` for Artificial Intelligence):

1.  **📊 Custom Student Monitoring:**
    * The instructor will exclusively see the students enrolled in their assigned subject and no one else.
    * The system will display attendance rates, GPA, and the automatic AI prediction status for each student.
2.  **🚨 Quick Intervention:**
    * When a student is in the danger zone (At Risk), a dedicated button named **"Send Auto Alert"** will appear for the instructor.
    * Clicking it sends an immediate, automated warning notification to that specific student's portal.
3.  **⭐ System Evaluation:**
    * The instructor can scroll to the bottom of the page, select a star rating, leave a comment, and submit it to instantly feed the Admin's evaluation table.

---

### 🎓 Third: Student Portal
A highly simplified portal that the student accesses by entering their academic ID only (e.g., `KGL1001`):

1.  **📈 My Academic Card:** The student views their current academic status and grades.
2.  **🔔 Notifications Box:** The student receives alerts directed to them by instructors, including the subject name and the date of the directive.
3.  **✍️ Feedback & Evaluation:** The student can submit a 5-point evaluation regarding their experience with the system.

---

## 🧪 5. Automated Testing System

To demonstrate professional development standards to the discussion committee, a complete and independent programmatic testing system is provided:
* **Responsible File:** `test_app.py`
* **Technologies Used:** `pytest`, `unittest.mock`
* **How to Run:** Open the command prompt and type the following direct command:
    ```bash
    pytest test_app.py -v
    ```
* This file will run **25 vital test cases** covering logins, duplicate emails, hacking attempts, and uploading incorrect file formats. It ensures all cases pass perfectly to prove software quality and security.

---

🌟 The program has been modified and fortified with all protection measures against incorrect data entry (Robust Validation). It is fully ready for the discussion and testing phase with immense confidence and a premium appearance fitting your high academic level! 🎓🚀✨