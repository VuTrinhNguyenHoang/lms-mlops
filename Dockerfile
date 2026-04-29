FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PROJECT_ROOT=/app \
    LOCAL_STORAGE_DIR=/app/storage \
    LOCAL_OUTPUT_DIR=/app/outputs \
    REFERENCE_DATA_PATH=/app/data/reference/simulated_data.csv \
    MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY scripts ./scripts
COPY README.md ./README.md

RUN mkdir -p /app/storage /app/outputs /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
