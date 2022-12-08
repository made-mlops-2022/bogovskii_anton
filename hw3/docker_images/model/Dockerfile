FROM python:3.10-slim
WORKDIR /opt/mlops/hw3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
ENTRYPOINT ["python3", "/opt/mlops/hw3/main.py"]
