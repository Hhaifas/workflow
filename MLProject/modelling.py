import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import argparse
# from preprocess import preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data_preprocessing.csv')
parser.add_argument('--test_size', type=float, default=0.3)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

# Menjalankan eksperimen agar disimpan pada Tracking UI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Menyimpan data hasil pelatihan pada satu pipeline
mlflow.set_experiment('membangun_model')

# Memuat dataset
# data = pd.read_csv('data_preprocessing.csv')
data = pd.read_csv(args.data_path)
input_example = data.drop('Personality', axis=1).iloc[0:8]
    
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('Personality', axis=1),
    data['Personality'],
    random_state=42,
    test_size=0.2
)

# Membuat sesi eksperimen baru untuk mencatat semua aktivitas terkait 

n_estimators = 100
max_depth = 10
mlflow.autolog()

# Train model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

mlflow.sklearn.log_model(
    sk_model = model, 
    artifact_path = "model",
    input_example = input_example
)


# Log metrics 
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)


# evaluasi f1-score
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro') 
mlflow.log_metric('f1_score', f1)