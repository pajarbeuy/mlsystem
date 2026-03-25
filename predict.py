import mlflow
import pandas as pd

logged_model = 'runs:/c7ccbcb01ae943328c32649fdf44c657/model'
# load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# predict on a Pandas DataFrame.
predict=loaded_model.predict(pd.DataFrame([[6.7, 3.1, 5.6, 2.4]]))
print(predict)