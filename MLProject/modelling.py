import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import f1_score
# from preprocess import preprocess_data
# import numpy as np
import dagshub
dagshub.init(repo_owner='Hhaifas', repo_name='membangun_model', mlflow=True)

# Menjalankan eksperimen agar disimpan pada Tracking UI
# konfigurasi mlflow di dagshub
mlflow.set_tracking_uri("https://dagshub.com/Hhaifas/membangun_model.mlflow")

# Menyimpan data hasil pelatihan pada satu pipeline
mlflow.set_experiment('membangun_model')

# Memuat dataset
data = pd.read_csv('MLProject/data_preprocessing.csv')
# input_example = data[0:8]

# Model online learning
model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000)
# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop('Personality', axis=1),
#     data['Personality'],
#     random_state=42,
#     test_size=0.2
# )

# Kelas target untuk partial_fit
classes = data['Personality'].unique()

# Membuat sesi eksperimen baru untuk mencatat semua aktivitas terkait 
with mlflow.start_run():
    mlflow.autolog()
    
    # Preprocessing data untuk batch
    X_batch = data.drop(columns=['Personality'])
    y_batch = data['Personality']
    
    # model = RandomForestClassifier(n_estimators=100, max_depth=10)
    # Initial fit unttuk batch pertama
    model.partial_fit(X_batch, y_batch, classes=classes)
    
    # Log metrics 
    accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    # Simpan model ke file lokal
    dump(model, "MLProject/model/membangun_model.joblib")
    
    # Log file model sebagai artifak ke MLflow
    mlflow.log_artifact("membangun_model.joblib", artifact_path="model_artifacts")
    
    
    # Log model setelah selesai training
    mlflow.sklearn.log_model(
        sk_model = model, 
        artifact_path = "membangun_model",
        input_example = X_batch.iloc[0:8]
    )
    
    # model.fit(X_train, y_train)
    
   
    
    # Metrik tambahan (manual logging)
    # waktu training
    start_time = time.time()
    model.fit(X_batch, y_batch)
    training_time = time.time() - start_time
    mlflow.log_metric('training_time', training_time)
    
    # evaluasi f1-score
    y_pred = model.predict(X_batch)
    f1 = f1_score(y_batch, y_pred, average='macro') 
    mlflow.log_metric('f1_score', f1)


# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import SGDClassifier
# import os
# import numpy as np
# import pandas as pd
# import warnings
# import sys

# import dagshub
# dagshub.init(repo_owner='Hhaifas', repo_name='membangun_model', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/Hhaifas/membangun_model.mlflow")

# # Menyimpan data hasil pelatihan pada satu pipeline
# mlflow.set_experiment('membangun_model')

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     np.random.seed(42)
    
#     loss = sys.argv[1] if len(sys.argv) > 1 else 'log_loss'
#     learning_rate = sys.argv[2] if len(sys.argv) > 2 else 'adaptive'
#     eta0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
#     max_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
#     # read dataset
#     filepath = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_preprocessing.csv")
#     data = pd.read_csv(filepath)
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         data.drop("Personality", axis=1),
#         data['Personality'],
#         random_state=42,
#         test_size=0.3
#     )
#     input_example=X_train[0:8]
        
#     with mlflow.start_run():
#         model = SGDClassifier(loss=loss, learning_rate=learning_rate,
#                               eta0=eta0, max_iter=max_iter)
#         model.fit(X_train, y_train)
        
#         # predicted_qualities = model.predict(X_test)
        
#         mlflow.sklearn.log_model(
#             sk_model=model,
#             artifact_path="model",
#             input_example=input_example
#         )
        
#         # Log metrics
#         accuracy = model.score(X_test, y_test)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_param("loss", loss)
#         mlflow.log_param("learning_rate", learning_rate)
#         mlflow.log_param("eta0", eta0)
#         mlflow.log_param("max_iter", max_iter)
