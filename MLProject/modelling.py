import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.linear_model import SGDClassifier
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import dagshub
dagshub.init(repo_owner='Hhaifas', repo_name='membangun_model', mlflow=True)

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Check if Dagshub credential are available
if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
    # Set up MLflow tracking with Dagshub
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')
    mlflow.set_tracking_uri("https://dagshub.com/Hhaifas/membangun_model.mlflow")
    mlflow.set_experiment('membangun_model')
    use_remote_tracking = True
    print("Using remote MLflow tracking on Dagshub")
else:
    # Use local tracking
    print("Dagshub credentials not found. Using local tracking")
    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("membangun_model_local")
    use_remote_tracking = False
    
# loda data
data = pd.read_csv(args.data_path)

X = data.drop(columns=['Personality'])
y = data['Personality']

X = pd.get_dummies(X, drop_first=True)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Auto logging
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = SGDClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Metrics
    accuracy = model.score(X_train, y_train)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X_train.shape[1])
    
    # save model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    mlflow.log_artifact(args.model_output)
    
    if use_remote_tracking:
        print("Lihat hasil tracking di Dagshub")
        print("https://dagshub.com/Hhaifas/membangun_model.mlflow")
    else:
        print("MLflow tersimpan secara lokal di ./mlruns")
        
    # Register the model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "membangun_model"
    
    try:
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"model registered as '{registered_model_name}'")
    except Exception as e:
        print(f"Failed to register model: {e}")
        
    # instruction for serving model
    if use_remote_tracking:
        print("to serve the model, user the following command:")
        print(f"mlflow models server -m 'models:/{registered_model_name}/laterst' --port 5000")
    else:
        print('to serve the model locally, use the follwoing command:')
        print(f"mlflow models serve -m '{model_uri}' --port 5000")


# Menjalankan eksperimen agar disimpan pada Tracking UI
# konfigurasi mlflow di dagshub
# mlflow.set_tracking_uri("https://dagshub.com/Hhaifas/membangun_model.mlflow")

# Menyimpan data hasil pelatihan pada satu pipeline
# mlflow.set_experiment('membangun_model')

# Memuat dataset
# data = pd.read_csv('data_preprocessing.csv')
# input_example = data[0:8]

# Model online learning
# model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000)
# # X_train, X_test, y_train, y_test = train_test_split(
# #     data.drop('Personality', axis=1),
# #     data['Personality'],
# #     random_state=42,
# #     test_size=0.2
# # )

# # Kelas target untuk partial_fit
# classes = data['Personality'].unique()

# # Membuat sesi eksperimen baru untuk mencatat semua aktivitas terkait 
# with mlflow.start_run():
#     mlflow.autolog()
    
#     # Preprocessing data untuk batch
#     X_batch = data.drop(columns=['Personality'])
#     y_batch = data['Personality']
    
#     # model = RandomForestClassifier(n_estimators=100, max_depth=10)
#     # Initial fit unttuk batch pertama
#     model.partial_fit(X_batch, y_batch, classes=classes)
    
#     # Log metrics 
#     accuracy = model.score(X_batch, y_batch)
#     mlflow.log_metric("accuracy", accuracy)
#     print(f"Accuracy: {accuracy:.4f}")
#     # Simpan model ke file lokal
#     dump(model, "membangun_model.joblib")
    
#     # Log file model sebagai artifak ke MLflow
#     mlflow.log_artifact("membangun_model.joblib", artifact_path="model_artifacts")
    
    
#     # Log model setelah selesai training
#     mlflow.sklearn.log_model(
#         sk_model = model, 
#         artifact_path = "membangun_model",
#         input_example = X_batch.iloc[0:8]
#     )
    
#     # model.fit(X_train, y_train)
    
   
    
#     # Metrik tambahan (manual logging)
#     # waktu training
#     start_time = time.time()
#     model.fit(X_batch, y_batch)
#     training_time = time.time() - start_time
#     mlflow.log_metric('training_time', training_time)
    
#     # evaluasi f1-score
#     y_pred = model.predict(X_batch)
#     f1 = f1_score(y_batch, y_pred, average='macro') 
#     mlflow.log_metric('f1_score', f1)
    