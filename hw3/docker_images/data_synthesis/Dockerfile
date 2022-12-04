FROM python:3.10-slim
WORKDIR /opt/mlops/hw3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY generate_data.py .
CMD python3 /opt/mlops/hw3/generate_data.py /data/raw/generated_dataset
