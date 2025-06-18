# Gunakan base image Python yang ringan
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements dan install dependencies
COPY MLProject/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy seluruh source code project ke image
COPY MLProject/ ./MLProject/
COPY data_preprocessing.csv .

# (Optional) Set environment variable agar output log langsung tampil
ENV PYTHONUNBUFFERED=1

# Default command: jalankan script python
ENTRYPOINT ["python", "MLProject/modelling.py"]