import mlflow
import json
from mlflow.models import validate_serving_input
from mlflow.models import convert_input_example_to_serving_input

# Run ID dari model yang sudah dijalankan
run_id = "c7ccbcb01ae943328c32649fdf44c657"
model_uri = f"runs:/{run_id}/model"

# Set tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Load model dari MLflow
print(f"Loading model dari: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# Baca input example dari artifacts
input_example_path = f"mlartifacts/0/{run_id}/artifacts/model/input_example.json"
with open(input_example_path, 'r') as f:
    input_example = json.load(f)

print(f"\nInput Example:")
print(json.dumps(input_example, indent=2))

# Konversi ke serving format
serving_payload = convert_input_example_to_serving_input(input_example)
print(f"\nServing Payload:")
print(json.dumps(serving_payload, indent=2))

# Validate serving input
try:
    validate_serving_input(model_uri, serving_payload)
    print("\n✓ Serving input validation successful!")
except Exception as e:
    print(f"\n✗ Validation error: {e}")