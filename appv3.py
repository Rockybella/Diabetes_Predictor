from fastapi import FastAPI, Header, HTTPException
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.post("/predict")
async def predict_diabetes(data: dict, x_token: str = Header(None)):
    # 1. Validate Secret Token (The hex16 you generated)
    if x_token != os.getenv("API_AUTH_TOKEN"):
        raise HTTPException(status_code=403, detail="Unauthorized")

    # 2. Extract data from request
    custom_data = CustomData(
        preg=data['preg'], plas=data['plas'], pres=data['pres'],
        skin=data['skin'], test=data['test'], bmi=data['bmi'],
        pedi=data['pedi'], age=data['age']
    )
    
    # 3. Run Pipeline
    df = custom_data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    return {"result": int(result[0])}
