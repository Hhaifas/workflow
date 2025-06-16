import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data
# import numpy as np


# Menjalankan eksperimen agar disimpan pada Tracking UI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Menyimpan data hasil pelatihan pada satu pipeline
mlflow.set_experiment('membangun_model')

# Memuat dataset
data = pd.read_csv('personality_dataset.csv')
input_example = data[0:8]
    
X_train, X_test, y_train, y_test = preprocess_data(
    data,
    'Personality',
    'preprocessor_pipeline.joblib',
    'data.csv'
)

# Membuat sesi eksperimen baru untuk mencatat semua aktivitas terkait 
with mlflow.start_run():
    n_estimators = 505
    max_depth = 37
    mlflow.autolog()
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model = model, 
        artifact_path = "model",
        input_example = input_example
    )
    
    model.fit(X_train, y_train)
    
    # Log metrics 
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)