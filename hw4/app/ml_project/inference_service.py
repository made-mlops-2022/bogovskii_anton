from prepare_model import BaselineModel

from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import joblib
import numpy as np
import os


app = FastAPI()
model = BaselineModel.load(os.environ['MODEL_PATH'])


class Vec(BaseModel):
    vals: List[float]


@app.get('/alive')
async def alive():
    pass


@app.get('/ready')
async def ready():
    pass


@app.post('/predict')
async def predict(value: Vec):
    return {
        'result': int(model.predict(np.fromiter(value.vals, dtype=float).reshape(1, -1)))
    }
