import os
import sys
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

def setup_mlflow(experiment_name="membangun_model"):
    """Setup MLflow tracking either local or remote (DagsHub) based on environment variable"""
    load_dotenv()
    dagshub_token = os.environ.get('DAGSHUB_TOKEN')
    dagshub_username = os.environ.get('DAGSHUB_USERNAME')
    if dagshub_token and dagshub_username:
        # Remote tracking (DagsHub)
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        tracking_uri = f"https://dagshub.com/{dagshub_username}/Personality-Tracking-Model.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ Using remote MLflow tracking on DagsHub: {tracking_uri}")
    else:
        # Local tracking
        os.makedirs('mlruns', exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        print("🏠 Using local MLflow tracking.")
    mlflow.set_experiment(experiment_name)

def preprocess_data(df):
    # Fill missing values with median (if any)
    df = df.copy()
    if df.isnull().any().any():
        df.fillna(df.median(numeric_only=True), inplace=True)
    # Separate features and target
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=args.random_state, test_size=args.test_size, stratify=y
    )
    # Handle imbalance if needed (optional, uncomment if imbalance)
    # oversampler = RandomOverSampler(random_state=args.random_state)
    # X_train, y_train = oversampler.fit_resample(X_train, y_train)
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_model_local(model, model_name="model_rf.pkl"):
    os.makedirs("model_output", exist_ok=True)
    out_path = os.path.join("model_output", model_name)
    import joblib
    joblib.dump(model, out_path)
    print(f"📁 Model juga disimpan ke {out_path}")
    return out_path

def main(args):
    setup_mlflow(experiment_name="membangun_model")
    # Load dataset
    if not os.path.exists(args.data_path):
        print(f"❌ File {args.data_path} tidak ditemukan.")
        sys.exit(1)
    data = pd.read_csv(args.data_path)
    input_example = data.drop('Personality', axis=1).iloc[[0]].to_dict(orient="records")[0]
    # Preprocess (with scaling & optional balancing)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    n_estimators = 100
    max_depth = 10
    mlflow.sklearn.autolog(log_input_examples=False)

    run_name = f"RandomForest_{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Log model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=[input_example],
        )
        print("✅ Model berhasil disimpan ke MLflow.")

        # Save model ke lokal juga untuk artifact
        save_model_local(model, model_name="model_rf.pkl")

        # Log metrics manual (autolog sudah log, tapi contoh manual)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        print(f"📊 Akurasi: {accuracy:.4f}, F1: {f1:.4f}")

        # Log info CI/CD jika dijalankan di GitHub Actions
        if os.environ.get('GITHUB_SHA'):
            mlflow.log_param("github_sha", os.environ['GITHUB_SHA'])
            mlflow.log_param("github_ref", os.environ.get('GITHUB_REF'))
            mlflow.log_param("github_run_number", os.environ.get('GITHUB_RUN_NUMBER'))

        # Info tracking
        tracking_mode = "remote" if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME') else "local"
        mlflow.log_param("tracking_mode", tracking_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_preprocessing.csv')
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    main(args)