FROM python:3.10-slim
WORKDIR /opt/mlops/hw3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY process.py split.py train.py predict.py train.sh predict.sh ./
