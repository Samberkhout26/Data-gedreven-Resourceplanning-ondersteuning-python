# Rister ML Pipeline
# Multi-stage build: dependencies apart van code zodat image-updates snel gaan

FROM python:3.11-slim AS base

# Systeem-dependencies voor geopandas (GDAL/GEOS) en Firebird
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libfbclient2 \
    firebird-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies eerst (apart layer zodat pip install niet herhaalt bij code-wijziging)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime stage ---
FROM base AS runtime

WORKDIR /app

# Broncode kopiëren
COPY src/ ./src/

# Data/models directories aanmaken (worden gemount of gevuld tijdens run)
RUN mkdir -p data/processed data/extern models/onnx

# Prefect worker als standaard entrypoint
# Kan overschreven worden via docker-compose of Container App command
CMD ["prefect", "worker", "start", "--pool", "rister-pool"]
