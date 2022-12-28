from fastapi.testclient import TestClient
from prepare_model import CATEGORY_INDS, load_dataset, train_model
import os


def test_predict():
    X, y = load_dataset('./data/raw/heart_cleveland_upload.csv')
    train_model('model.joblib', X, y, CATEGORY_INDS)
    os.environ['MODEL_PATH'] = 'model.joblib'

    from inference_service import app
    client = TestClient(app)

    response = client.post('predict', json={'vals': [69,1,0,160,234,1,2,131,0,0.1,1,1,1]})
    assert response.status_code == 200
    assert response.json() == {'result': 1}
