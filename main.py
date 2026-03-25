import mlflow
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set environment variables untuk MLflow local storage
os.environ["MLFLOW_BACKEND_STORE_URI"] = "./mlruns"
os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "./mlruns/artifacts"

# Muat dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logging dengan MLflow
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Bangun dan latih model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi dan hitung akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    print(f"\nModel Training Complete!")
    print(f"Accuracy: {accuracy:.4f}")