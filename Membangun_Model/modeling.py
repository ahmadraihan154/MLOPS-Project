import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings(action='ignore')

'''
catatan :
Autolog tidak digunakan karena MLflow autolog tidak bisa membaca model dalam kasus ini 
karena ada proses transformasi inverse_transform pada target (y_test dan y_pred) yang tidak 
dikenali secara otomatis oleh autolog()
'''

# Load Data
data_path = r'D:\6. Membangun Sistem Machine Learning\Membangun_Model\diamond_preprocessing'
transformer = joblib.load(os.path.join(data_path, 'power_transformers.joblib'))
price_transformer = transformer['price']

X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
y_test = price_transformer.inverse_transform(y_test.to_numpy().reshape(-1,1))

# Tentukan nama experimen dan hasil experimennya
mlflow.set_tracking_uri("http://127.0.0.1:5000") # jangan lupa di terminal ketik mlflow ui
mlflow.set_experiment('base-model_experiment')

# Melakukan Experimen pelatihan model
with mlflow.start_run(run_name='LGBM_Base'):
    # Melatih Model
    model = LGBMRegressor()
    model.fit(X_train, y_train)

    # Mencatat performa model
    y_pred_transform = model.predict(X_test)
    y_pred = price_transformer.inverse_transform(y_pred_transform.reshape(-1,1))
    r2_skor = r2_score(y_test, y_pred)
    rmse_skor = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_params(model.get_params())
    mlflow.log_metric("RMSE", rmse_skor)    
    mlflow.log_metric("R2", r2_skor)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f'R2 Score : {r2_skor}')
    print(f'RMSE : {rmse_skor}')