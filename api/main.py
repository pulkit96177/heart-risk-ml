from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np

# ── models ────────────────────────────────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models['pipeline'] = joblib.load('../models/lr_best_model.pkl')
    ml_models['kmeans']   = joblib.load('../models/kmeans_model.pkl')
    ml_models['scaler']   = joblib.load('../models/scaler.pkl')
    yield

# ── app ───────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── input schema ──────────────────────────────────────────────
class PatientInput(BaseModel):
    age: float
    sex: str
    trestbps: float
    chol: float
    fbs: str
    thalch: float
    exang: str
    oldpeak: float
    cp: str
    restecg: str
    slope: str
    dataset: str

# ── preprocessing ─────────────────────────────────────────────
def preprocess(data: PatientInput) -> pd.DataFrame:
    # binary
    sex   = 1 if data.sex   == "Male" else 0
    fbs   = 1 if data.fbs   == "Yes"  else 0
    exang = 1 if data.exang == "Yes"  else 0

    # one hot — cp
    cp_asymptomatic   = 1 if data.cp == "asymptomatic"    else 0
    cp_atypical       = 1 if data.cp == "atypical angina" else 0
    cp_non_anginal    = 1 if data.cp == "non-anginal"     else 0
    cp_typical        = 1 if data.cp == "typical angina"  else 0

    # one hot — restecg
    restecg_lv        = 1 if data.restecg == "lv hypertrophy"    else 0
    restecg_normal    = 1 if data.restecg == "normal"             else 0
    restecg_st        = 1 if data.restecg == "st-t abnormality"  else 0

    # one hot — slope
    slope_down        = 1 if data.slope == "downsloping" else 0
    slope_flat        = 1 if data.slope == "flat"        else 0
    slope_up          = 1 if data.slope == "upsloping"   else 0

    # one hot — dataset
    ds_cleveland      = 1 if data.dataset == "Cleveland"      else 0
    ds_hungary        = 1 if data.dataset == "Hungary"        else 0
    ds_switzerland    = 1 if data.dataset == "Switzerland"    else 0
    ds_va             = 1 if data.dataset == "VA Long Beach"  else 0

    # build row without cluster first
    row = {
        'age':                        data.age,
        'sex':                        sex,
        'trestbps':                   data.trestbps,
        'chol':                       data.chol,
        'fbs':                        fbs,
        'thalch':                     data.thalch,
        'exang':                      exang,
        'oldpeak':                    data.oldpeak,
        'dataset_Cleveland':          ds_cleveland,
        'dataset_Hungary':            ds_hungary,
        'dataset_Switzerland':        ds_switzerland,
        'dataset_VA Long Beach':      ds_va,
        'cp_asymptomatic':            cp_asymptomatic,
        'cp_atypical angina':         cp_atypical,
        'cp_non-anginal':             cp_non_anginal,
        'cp_typical angina':          cp_typical,
        'restecg_lv hypertrophy':     restecg_lv,
        'restecg_normal':             restecg_normal,
        'restecg_st-t abnormality':   restecg_st,
        'slope_downsloping':          slope_down,
        'slope_flat':                 slope_flat,
        'slope_upsloping':            slope_up,
    }

    df = pd.DataFrame([row])

    # assign cluster using kmeans on scaled data
    df_scaled  = ml_models['scaler'].transform(df)
    cluster    = ml_models['kmeans'].predict(df_scaled)[0]
    df['cluster'] = int(cluster)

    return df

# ── health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── predict ───────────────────────────────────────────────────
@app.post("/predict")
def predict(data: PatientInput):
    try:
        df          = preprocess(data)
        probability = ml_models['pipeline'].predict_proba(df)[0][1]
        prediction  = int(probability >= 0.5)
        risk_label  = "High Risk" if prediction == 1 else "Low Risk"

        return {
            "prediction":  prediction,
            "probability": round(float(probability), 3),
            "risk_label":  risk_label,
            "cluster":     int(df['cluster'].iloc[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))