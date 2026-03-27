import os
from pathlib import Path

# Root van het project (één niveau boven src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
EXTERN_DIR = DATA_DIR / "extern"
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
MODELS_ONNX_DIR = MODELS_DIR / "onnx"

# Database
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://samberkhout@localhost:5432/rister_prod17",
)
FIREBIRD_HOST = os.getenv("FIREBIRD_HOST", "mac-mini-van-terra.local:3050")
FIREBIRD_USER = os.getenv("FIREBIRD_USER", "SYSDBA")
FIREBIRD_PASSWORD = os.getenv("FIREBIRD_PASSWORD", "masterkey")

# MLflow
# In Azure ML wordt MLFLOW_TRACKING_URI automatisch gezet door de runtime.
# Lokaal valt het terug op de lokale server.
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_URI", "http://127.0.0.1:5001"))

# Azure ML
AML_SUBSCRIPTION_ID = os.getenv("AML_SUBSCRIPTION_ID", "")
AML_RESOURCE_GROUP = os.getenv("AML_RESOURCE_GROUP", "rister-ml")
AML_WORKSPACE = os.getenv("AML_WORKSPACE", "rister-aml")

# Externe bestanden
BAG_COORDS_PATH = EXTERN_DIR / "bag_nederland_coords.csv"
BODEM_GPKG_PATH = EXTERN_DIR / "Bodemkundige_Grondsoortenkaart_2025.gpkg"
