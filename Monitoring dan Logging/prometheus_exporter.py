from fastapi import FastAPI, Request, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import psutil
import logging
import time
import os
import mlflow

app = FastAPI()
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset and Preprocess")
MODEL_PATH = os.path.join(BASE_DIR, "mlruns", "566645058033808850", "dadcc406329f4f04a42f9d9df48d414c", "artifacts", "model")

# Model and artifacts
model = mlflow.pyfunc.load_model(MODEL_PATH)
encoder = joblib.load(os.path.join(DATA_DIR, "onehot_encoder.joblib"))
transformer = joblib.load(os.path.join(DATA_DIR, "power_transformers.joblib"))
price_transformer = transformer['price']

# Training data
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train_raw = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
y_train = price_transformer.inverse_transform(y_train_raw.values.reshape(-1,1)).flatten()

# Feature columns
numeric_cols = ['carat', 'x', 'y', 'z', 'table']
categoric_cols = ['cut', 'color', 'clarity']

# Metrics setup
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage (%)")
RAM_USAGE = Gauge("system_ram_usage_percent", "RAM usage (%)")
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk usage (%)")
PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions")
ERROR_COUNTER = Counter("model_errors_total", "Total errors")
HTTP_REQUESTS_COUNTER = Counter("http_requests_total", "Total HTTP requests")
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')
R2_SCORE = Gauge("model_r2_score", "R2 Score")
RMSE_SCORE = Gauge("model_rmse", "RMSE")
MAE_SCORE = Gauge("model_mae", "Mean Absolute Error")

@app.middleware("http")
async def count_requests(request: Request, call_next):
    HTTP_REQUESTS_COUNTER.inc()
    response = await call_next(request)
    return response

@app.get("/")
async def health_check():
    return {
        "status": "ready",
        "metrics": "available at /metrics",
        "predict": "POST JSON to /predict"
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    PREDICTION_COUNTER.inc()
    
    try:
        # System metrics
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage("/").percent)

        # Process input
        input_data = await request.json()
        input_df = pd.DataFrame([input_data])

        # Feature transformations
        numeric_features = []
        for col in numeric_cols:
            transformed = transformer[col].transform(input_df[[col]])
            numeric_features.append(transformed)
        
        scaled_features = np.concatenate(numeric_features, axis=1)
        encoded = encoder.transform(input_df[categoric_cols])
        processed_input = np.concatenate([encoded, scaled_features], axis=1)

        # Prediction
        prediction_scaled = model.predict(processed_input)
        predicted_price = price_transformer.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

        # Model performance metrics
        y_all_pred = np.append(y_train_pred, predicted_price)
        y_all_true = np.append(y_train, predicted_price)
        r2 = r2_score(y_all_true, y_all_pred)
        rmse = np.sqrt(mean_squared_error(y_all_true, y_all_pred))
        mae = mean_absolute_error(y_all_true, y_all_pred)
        
        R2_SCORE.set(r2)
        RMSE_SCORE.set(rmse)
        MAE_SCORE.set(mae)

        return {
            "predicted_price": float(predicted_price),
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae
        }

    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)

# Initialize training predictions
y_train_pred = None
def prepare_train_predictions():
    global y_train_pred
    y_train_pred_scaled = model.predict(X_train.values)
    y_train_pred = price_transformer.inverse_transform(y_train_pred_scaled.reshape(-1,1)).flatten()

prepare_train_predictions()