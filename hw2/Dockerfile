FROM python:3.10-slim
WORKDIR /hw2
ENV DATASET_PATH=/hw2/dataset.csv
ENV MODEL_PATH=/hw2/model.joblib
COPY ./data/raw/heart_cleveland_upload.csv /hw2/dataset.csv
COPY ./requirements_docker.txt /hw2/requirements.txt
RUN pip install -r /hw2/requirements.txt
COPY ./ml_project/*.py /hw2/
RUN python prepare_model.py $DATASET_PATH $MODEL_PATH && rm $DATASET_PATH
CMD uvicorn inference_service:app
