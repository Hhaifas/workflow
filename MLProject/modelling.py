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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import warnings 

warnings.filterwarnings("ignore")


class PersonalityCICD:
    def __init__(self, experiment_name="membangun_model"):
        self.experiment_name = experiment_name
        self.use_remote_tracking = False
        self.setup_environment()
        self.setup_mlflow()
        
    def setup_environment(self):
        load_dotenv()
        print("Setting up environment configuration...")
        
        # check if required environment
        dagshub_token = os.environ.get('DAGSHUB_TOKEN')
        dagshub_username = os.environ.get('DAGSHUB_USERNAME')
        
        if dagshub_token and dagshub_username:
            print("Dagshub credentials found in environment")
            self.use_remote_tracking = True
        else:
            print("Dagshub credentials NOT found in environment")
            self.use_remote_tracking = False
    
    def setup_mlflow(self):
        if self.use_remote_tracking:
            if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
                os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME')
                os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')
                
                dagshub_username = os.environ.get('DAGSHUB_USERNAME')
                tracking_uri = f"https://dagshub.com/{dagshub_username}/membangun_model.mlflow"
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                
                print("Using remote MLFLOW tracking on Dagshub")
                print(f"Tracking URI: {tracking_uri}")
                print(f"Username: {dagshub_username}")
            else:
                self.use_remote_tracking = False
                self.setup_local_tracking()
        else:
            self._setup_local_tracking()
    
    def _setup_local_tracking(self):
        # Use local tracking if credentials aren't available
        print("🏠 DagsHub credentials not found. Using local MLflow tracking")
        os.makedirs('mlruns', exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        print("📁 Local MLflow tracking initialized")
        
    def create_sample_data(self):
        print("Creating sample data...")
        
        np.random.seed(42)
        n_samples=100
        
        data = {
            'Time_spent_Alone': np.random.normal(4, 3, n_samples),
            'Stage_fear': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'Social_event_attendance': np.random.normal(3, 2, n_samples),
            'Going_outside': np.random.normal(3, 2, n_samples),
            'Drained_after_socializing': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'Friends_circle_size': np.random.normal(6, 4, n_samples),
            'Post_frequency': np.random.normal(3, 2, n_samples),
            'Personality': np.random.choice(['Introvert', 'Extrovert'], n_samples, p=[0.5, 0.5])
        }
        
        df = pd.DataFrame(data)
        
        missing_rate = 0.1  # Misal 10% missing value per kolom

        for col in df.columns:
            n_missing = int(n_samples * missing_rate)
            missing_indices = np.random.choice(n_samples, size=n_missing, replace=False)
            df.loc[missing_indices, col] = np.nan
        
        return df
    
    def load_and_preprocess_data(self, data_path):
        if os.path.exists(data_path):
            print(f"Loading dataset from {data_path}..")
            df = pd.read_csv(data_path)
        else:
            print(f"File {data_path} not found. creating sample data...")
            df = self.create_sample_data()
            
        # 2. Identify categorical and numerical columns
        categorical = df.select_dtypes(include=['object']).columns
        numerical = df.select_dtypes(include=['number']).columns

        # 3. Impute missing values
        # Numerical: mean
        imputer_num = SimpleImputer(strategy='mean')
        df[numerical] = imputer_num.fit_transform(df[numerical])
        # Categorical: 'missing'
        imputer_cat = SimpleImputer(strategy='constant', fill_value='missing')
        df[categorical] = imputer_cat.fit_transform(df[categorical])

        # 4. Drop duplicates
        df = df.drop_duplicates()

        # 5. Standardize numerical features
        scaler = StandardScaler()
        df[numerical] = scaler.fit_transform(df[numerical])

        # 6. Encode categorical features
        encoder = OrdinalEncoder()
        df[categorical] = encoder.fit_transform(df[categorical])

        # 7. Split features and target
        X = df.drop('Personality', axis=1)
        y = df['Personality']

        # 8. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, scaler, encoder
    
    def train_model(self, model, model_name, X_train, X_test, y_train, y_test):
        # Train model with MLflow tracking
        run_name = f"{model_name}_CI_{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"🚀 Training {model_name} in CI/CD pipeline...")
            
            # Enable autologging
            mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=True)
            
            # Log CI/CD specific parameters
            mlflow.log_param("pipeline_type", "CI/CD")
            mlflow.log_param("environment", "docker" if os.environ.get('DOCKER_ENV') else "local")
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("tracking_mode", "remote" if self.use_remote_tracking else "local")
            
            # Log GitHub Actions info if available
            if os.environ.get('GITHUB_SHA'):
                mlflow.log_param("github_sha", os.environ.get('GITHUB_SHA'))
                mlflow.log_param("github_ref", os.environ.get('GITHUB_REF'))
                mlflow.log_param("github_run_number", os.environ.get('GITHUB_RUN_NUMBER'))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Log additional metrics
            mlflow.log_metric("ci_accuracy", accuracy)
            mlflow.log_metric("ci_f1_score", f1)
            mlflow.log_metric("ci_precision", precision)
            mlflow.log_metric("ci_recall", recall)
            
            # Create model output directory
            os.makedirs("model_output", exist_ok=True)
            
            # Save model locally for artifact upload
            model_path = f"model_output/{model_name}_model.pkl"
            mlflow.sklearn.save_model(model, model_path)
            
            print(f"✅ {model_name} CI training completed!")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   📊 F1 Score: {f1:.4f}")
            print(f"   🔄 Tracking: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
            
            return model, accuracy, f1
        
    def run_ci_pipeline(self, data_path):
        # Run complete CI/CD pipeline
        print("🔄 Starting CI/CD Pipeline...")
        print(f"🌐 Environment: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
        
        # Load data
        X_train, X_test, y_train, y_test, scaler, encoder = self.load_and_preprocess_data(data_path)
        
        # Pakai satu model saja, misal RandomForest
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        model_name = "RandomForest"
        
        try:
            trained_model, accuracy, f1 = self.train_model(
                model, model_name, X_train, X_test, y_train, y_test
            )
            results = {
                model_name: {
                    'model': trained_model,
                    'accuracy': accuracy,
                    'f1_score': f1
                }
            }
            best_model = model_name
            best_score = f1
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            results = {}
            best_model = None
            best_score = 0
        
        # Log pipeline summary (seperti sebelumnya)
        summary_run_name = f"CI_Pipeline_Summary_{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
        with mlflow.start_run(run_name=summary_run_name):
            mlflow.log_param("total_models_trained", len(results))
            mlflow.log_param("best_model", best_model)
            mlflow.log_param("tracking_mode", "remote" if self.use_remote_tracking else "local")
            mlflow.log_metric("best_f1_score", best_score)
            mlflow.log_param("dagshub_username", os.environ.get('DAGSHUB_USERNAME', 'not_set'))
            mlflow.log_param("docker_username", os.environ.get('DOCKER_HUB_USERNAME', 'not_set'))
            for name, result in results.items():
                mlflow.log_metric(f"{name}_accuracy", result['accuracy'])
                mlflow.log_metric(f"{name}_f1_score", result['f1_score'])

        print(f"\n🎉 CI/CD Pipeline completed!")
        print(f"📊 Best model: {best_model} (F1: {best_score:.4f})")
        print(f"📁 Model artifacts saved to: model_output/")
        print(f"🔄 Tracking mode: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
        
        return results
    
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Personality CI/CD')
    parser.add_argument('--data_path', type=str, default='data_preprocessing.csv',
                       help='Path to the dataset')
    parser.add_argument('--experiment_name', type=str, default='membangun_model',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    try:
        # Initialize CI/CD pipeline
        pipeline = PersonalityCICD(experiment_name=args.experiment_name)
        
        # Run pipeline
        results = pipeline.run_ci_pipeline(args.data_path)
        
        print("\n🎯 CI/CD Pipeline execution successful!")
        
    except Exception as e:
        print(f"❌ CI/CD Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()