# Gunakan base image Python yang ringan dan kompatibel
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Copy requirements file dan install dependencies
COPY MLProject/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy semua source code project ke dalam image
COPY MLProject/ ./MLProject/
COPY data_preprocessing.csv .
COPY scripts/ ./scripts/

# Pastikan folder output model ada
RUN mkdir -p model_output

# Set agar output log Python langsung tampil
ENV PYTHONUNBUFFERED=1

# Default command: jalankan script modelling.py
ENTRYPOINT ["python", "MLProject/modelling.py"]