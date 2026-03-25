import mlflow.pyfunc
import pandas as pd

# URI model yang di registered
model_uri = "models:/random_forest@test_val"

# load model
model = mlflow.pyfunc.load_model(model_uri)

# gunakan model untuk prediksi
input_data = pd.DataFrame([[6.7, 3.1, 5.6, 2.4]], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
prediksi = model.predict(input_data)
print(prediksi)