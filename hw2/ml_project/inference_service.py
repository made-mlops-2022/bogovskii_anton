from prepare_model import BaselineModel

from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np
import os


app = FastAPI()
model = None


class Vec(BaseModel):
    vals: list[float]


@app.on_event("startup")
async def startup_event():
    global model
    model = BaselineModel.load(os.environ['MODEL_PATH'])


@app.get('/health')
async def health():
    assert model is not None


@app.post('/predict')
async def predict(value: Vec):
    return {
        'result': int(model.predict(np.fromiter(value.vals, dtype=float).reshape(1, -1)))
    }
