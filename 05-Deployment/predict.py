import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI(title="Lead-Score-Conversion")

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


@app.post("/predict")
def predict(lead:Lead):
    client = lead.model_dump()
    X = [client]
    probability = pipeline.predict_proba(X)[0, 1]
    return {"probability": float(probability)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
