import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Load the model + DictVectorizer
with open("model.bin", "rb") as f_in:
    dv, rf = pickle.load(f_in)


# 2. Define the input schema (Pydantic model)
class InsuranceClient(BaseModel):
    KIDSDRIV: int
    AGE: float
    HOMEKIDS: int
    YOJ: float | None = None
    INCOME: float | None = None
    HOME_VAL: float | None = None
    PARENT1: str
    MSTATUS: str
    GENDER: str
    EDUCATION: str
    OCCUPATION: str
    TRAVTIME: int
    CAR_USE: str
    BLUEBOOK: float | None = None
    TIF: int
    CAR_TYPE: str
    RED_CAR: str
    OLDCLAIM: float | None = None
    REVOKED: str
    MVR_PTS: int
    CAR_AGE: float | None = None
    URBANICITY: str


# 3. Create the FastAPI app
app = FastAPI(title="Auto Insurance Claim Prediction API")


@app.get("/")
def home():
    return {"message": "Auto Insurance Claim Prediction API is running"}


# 4. Prediction endpoint
@app.post("/predict")
def predict_claim_probability(client: InsuranceClient):

    # Convert Pydantic model to dict
    client_data = client.model_dump()

    # DictVectorizer expects a list of records
    X = dv.transform([client_data])

    # Predict probability of claim = 1
    y_pred = rf.predict_proba(X)[0, 1]

    return {
        "claim_probability": float(y_pred),
        "will_likely_claim": bool(y_pred >= 0.5),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


