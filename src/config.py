"""Gedeelde configuratie voor de Rister ML pipeline."""

import os
from pathlib import Path

# ── Paden ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BAG_CSV_PATH = DATA_DIR / "bag_nederland_coords.csv"

# ── Database ──────────────────────────────────────────────────────────────────
POSTGRES_URI = os.environ.get("POSTGRES_URI", "")
FIREBIRD_HOST = os.environ.get("FIREBIRD_HOST", "mac-mini-van-terra.local:3050")
FIREBIRD_USER = os.environ.get("FIREBIRD_USER", "SYSDBA")
FIREBIRD_PASSWORD = os.environ.get("FIREBIRD_PASSWORD", "masterkey")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "rister-lightgbm-v1-plushoeveelheid_inc_finetune"

# ── Azure ML ──────────────────────────────────────────────────────────────────
AML_SUBSCRIPTION_ID = os.environ.get("AML_SUBSCRIPTION_ID", "")
AML_RESOURCE_GROUP = os.environ.get("AML_RESOURCE_GROUP", "rister-ml")
AML_WORKSPACE = os.environ.get("AML_WORKSPACE", "rister-aml")

# ── Features ──────────────────────────────────────────────────────────────────
CATEGORICAL = [
    "URENVERANTW_MEDID",
    "BEWERKING_ID",
    "DIENST_ART_ID",
    "RELATIE_ID",
    "REL_POSTCODE",
    "DIENST_ART_OMS",
    "MACH_OMS",
    "con",
    "bron",
    "EquipmentGroupTypes",
    "planninggroupsname",
]

NUMERICAL = [
    "lat",
    "lon",
    "dag_sin",
    "dag_cos",
    "maand_sin",
    "maand_cos",
    "week_sin",
    "week_cos",
    "med_std_tijd",
    "med_aantal_opdrachten",
    "med_ervaring_bewerking",
    "med_gem_tijd",
    "taak_gem",
    "med_klant_bezoeken",
    "med_klant_ratio",
    "med_klant_snelheid",
    "med_bewerking_snelheid",
    "med_klant_gem_tijd",
    "med_bewerking_gem_tijd",
    "med_totaal_opdrachten",
    "hoeveelheid_volume",
    "hoeveelheid_gewicht",
    "hoeveelheid_stuks",
    "hoeveelheid_aanwezig",
    "hoeveelheid_baal",
]

FEATURES = CATEGORICAL + NUMERICAL
TARGET_TIME = "REAL_WORKED_TIME"
TARGET_RANK = "suitability_score"

# ── Kolom mapping (Rister → WerkExpert namen) ────────────────────────────────
KOLOM_MAPPING = {
    "EmployeeId": "URENVERANTW_MEDID",
    "ServiceLineId_x": "DIENST_ART_ID",
    "PlanningGroupId": "BEWERKING_ID",
    "ClientId": "RELATIE_ID",
    "servicelinesname": "DIENST_ART_OMS",
    "equipmentname": "MACH_OMS",
    "PostalCode": "REL_POSTCODE",
    "lat": "lat",
    "lon": "lon",
    "Datum": "URENVERANTW_DATUM",
    "dag_sin": "dag_sin",
    "dag_cos": "dag_cos",
    "maand_sin": "maand_sin",
    "maand_cos": "maand_cos",
    "week_sin": "week_sin",
    "week_cos": "week_cos",
    "med_gem_tijd": "med_gem_tijd",
    "med_std_tijd": "med_std_tijd",
    "med_aantal_opdrachten": "med_aantal_opdrachten",
    "med_ervaring_bewerking": "med_ervaring_bewerking",
    "taak_gem": "taak_gem",
    "med_klant_bezoeken": "med_klant_bezoeken",
    "med_klant_ratio": "med_klant_ratio",
    "med_klant_snelheid": "med_klant_snelheid",
    "med_bewerking_snelheid": "med_bewerking_snelheid",
    "med_klant_gem_tijd": "med_klant_gem_tijd",
    "med_bewerking_gem_tijd": "med_bewerking_gem_tijd",
    "med_totaal_opdrachten": "med_totaal_opdrachten",
    "hoeveelheid_big_bag": "hoeveelheid_volume",
    "hoeveelheid_dag": "hoeveelheid_gewicht",
    "hoeveelheid_hectare": "hoeveelheid_stuks",
    "hoeveelheid_kubieke_meter": "hoeveelheid_aanwezig",
    "TenantId_x": "con",
    "hoeveelheid_uur": "REAL_WORKED_TIME",
}

RISTER_ONLY = ["EquipmentGroupTypes", "planninggroupsname", "hoeveelheid_baal"]
