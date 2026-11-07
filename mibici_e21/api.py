import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn

app = FastAPI(
    title="MiBici Demand Prediction API",
    description="API REST para predecir la demanda de bicicletas por estación usando MLflow.",
    version="1.0"
)

model = None

class TripRequest(BaseModel):
    station_id: int = Field(..., description="ID numérico de la estación de MiBici")
    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Día de la semana (0=Lunes, 6=Domingo)")
    is_weekend: int = Field(..., ge=0, le=1, description="Es fin de semana (0=No, 1=Sí)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "station_id": 52,
                "hour": 8,
                "day_of_week": 1,
                "is_weekend": 0
            }]
        }
    }

class TripResponse(BaseModel):
    station_id: int
    predicted_departures: float

@app.on_event("startup")
def load_model():
    global model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        model_uri = "models:/mibici_p1_demand/1"
        print(f"Descargando modelo desde MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("¡Modelo cargado exitosamente!")
    except Exception as e:
        print(f"Error crítico al cargar el modelo: {e}")

@app.post("/predict", response_model=TripResponse)
def predict_demand(request: TripRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    
    input_data = pd.DataFrame([request.model_dump()])
    
    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")
    
    return TripResponse(
        station_id=request.station_id,
        predicted_departures=float(prediction)
    )