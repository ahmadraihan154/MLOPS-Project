import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import dagshub
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

# Konfigurasi awal
warnings.filterwarnings("ignore")
os.makedirs("model", exist_ok=True)  

# ========== 1. INISIALISASI MLFLOW & DAGSHUB ==========
dagshub.init(
    repo_owner="ahmadraihan154",
    repo_name="membangun_sistem_machine_learning_project",
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/ahmadraihan154/membangun_sistem_machine_learning_project.mlflow")
mlflow.set_experiment("diamond-price-prediction")

# ========== 2. LOAD DATA & TRANSFORMER ==========
data_path = r"D:\6. Membangun Sistem Machine Learning\Membangun_Model\diamond_preprocessing"
transformer = joblib.load(os.path.join(data_path, "power_transformers.joblib"))
price_transformer = transformer["price"]

X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
y_test = price_transformer.inverse_transform(y_test.to_numpy().reshape(-1, 1))

# ========== 3. MODEL TRAINING & TUNING ==========
model = LGBMRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [-1, 20],
}

grid_model = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# ========== 4. MLFLOW RUN & LOGGING ==========
with mlflow.start_run(run_name="LGBM_GridSearch_Full") as run:
    # 4.1 Training Model
    start_time = time.time()
    grid_model.fit(X_train, y_train)
    best_model = grid_model.best_estimator_
    y_pred_transform = best_model.predict(X_test)
    y_pred = price_transformer.inverse_transform(y_pred_transform.reshape(-1, 1))
    training_time = time.time() - start_time

    # 4.2 Calculate Metrics
    r2_skor = r2_score(y_test, y_pred)
    rmse_skor = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_skor = mean_absolute_error(y_test, y_pred)

    # 4.3 Log Parameters & Metrics
    mlflow.log_params(grid_model.best_estimator_.get_params())
    mlflow.log_metrics({
        "R2": r2_skor,
        "RMSE": rmse_skor,
        "MAE": mae_skor,
        "Training_Time_Seconds": training_time
    })

    # 4.4 Save Model
    mlflow.sklearn.log_model(best_model, "model")

    # ========== 5. GENERATE ADDITIONAL ARTIFACTS ==========
    
    # 5.1 Metric Info (metric_info.json)
    metric_info = {
        "best_params": grid_model.best_params_,
        "metrics": {
            "R2": r2_skor,
            "RMSE": rmse_skor,
            "MAE": mae_skor,
        }
    }
    with open("model/metric_info.json", "w") as f:
        json.dump(metric_info, f)
    
    # 5.2 Prediction vs Actual Plot (prediction_vs_real.png)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='prediction vs actual')
    plt.plot(y_test, y_test, color='red', label='Ideal Line')
    plt.legend()
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Diamond Prices")
    plt.savefig("model/prediction_vs_real.png")
    plt.close()

    # 5.4 Log All Artifacts
    mlflow.log_artifacts("model")

    print(f"Training selesai Best Model: {grid_model.best_params_} R2 Score: {r2_skor:.4f} RMSE: {rmse_skor:.4f} Waktu Training: {training_time:.2f} detik")