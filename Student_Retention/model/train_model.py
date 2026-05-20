import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall

def train_ai(csv_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'student_retention_model.h5')

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": "The dataset was not found!"}

    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['retention'] = np.where((df['gpa'] < 2.0) | (df['attendance'] < 60), 1, 0)

    features = ['gpa', 'attendance', 'quiz_score', 'assignment_score']
    
    # Ensure columns exist
    if not all(col in df.columns for col in features):
        return {"error": "CSV is missing required features (gpa, attendance, quiz_score, assignment_score)"}

    X = df[features].values
    y = df['retention'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, len(features))),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

    model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), verbose=0)

    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    model.save(MODEL_PATH)
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "status": "success"
    }

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'student_data.csv')
    print("Training model from script...")
    results = train_ai(CSV_PATH)
    print(results)

if __name__ == "__main__":
    main()