from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app = FastAPI()

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_uri = 'models:/tracking-quickstart/3'
model = mlflow.pyfunc.load_model(model_uri)

class PredictionRequest(BaseModel):
    input_data: list 

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data = np.array(request.input_data).astype(np.float64)
        prediction = model.predict(data).tolist()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

class UpdateModelRequest(BaseModel):
    version: str

@app.post("/update-model")
async def update_model(request: UpdateModelRequest):
    global model
    try:
        model_uri = f"runs:/{request.version}/iris_model"
        model = mlflow.pyfunc.load_model(model_uri)
        return {"detail": "Model updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model update failed: {str(e)}")

