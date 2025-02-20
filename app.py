from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# Initialize FastAPI app
app = FastAPI()

# Constants
MODEL_PATH = "model.xgb"

# Load pre-trained model
if os.path.exists(MODEL_PATH):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
else:
    raise RuntimeError("Model file not found. Train and save the model first.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an Excel file for prediction.
    """
    try:
        # Load and process the Excel file
        contents = await file.read()
        df = pd.read_excel(contents, header=None)
        dataset = df.values

        # Use the last row as input for prediction
        last_row = dataset[-1, :].reshape(1, -1)

        # Perform prediction
        predictions = model.predict(last_row)

        # Round predictions to integers
        predictions = np.round(predictions).astype(int)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Basic health check for the API.
    """
    return {"message": "XGBoost Model Prediction API is running."}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)