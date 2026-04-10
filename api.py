# api.py

import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix

# ---- load model artifacts ----
MODEL_DIR = "models"

clf = joblib.load(f"{MODEL_DIR}/expense_classifier.pkl")
tfidf_word = joblib.load(f"{MODEL_DIR}/tfidf_word.pkl")
tfidf_char = joblib.load(f"{MODEL_DIR}/tfidf_char.pkl")
ohe = joblib.load(f"{MODEL_DIR}/ohe_vendor.pkl")
scaler = joblib.load(f"{MODEL_DIR}/amount_scaler.pkl")
le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

# ---- app ----
app = FastAPI(title="Expense Classifier API")


# ---- request schema ----
class Request(BaseModel):
    vendorId: str
    itemName: str
    itemDescription: str
    itemTotalAmount: float


# ---- feature pipeline ----
def build_features(data):
    name = data["itemName"].lower().strip()
    desc = data["itemDescription"].lower().strip()

    text = f"{name} {desc}" if desc and desc != name else name

    X_word = tfidf_word.transform([text])
    X_char = tfidf_char.transform([text])
    X_vendor = ohe.transform([[data["vendorId"]]])

    amt = np.array([[data["itemTotalAmount"], np.log1p(abs(data["itemTotalAmount"]))]])

    X_amt = csr_matrix(scaler.transform(amt))

    return hstack([X_word, X_char, X_vendor, X_amt])


# ---- prediction endpoint ----
@app.post("/predict")
def predict(req: Request):
    X = build_features(req.dict())
    pred = clf.predict(X)[0]

    return {"accountName": le.inverse_transform([pred])[0]}
