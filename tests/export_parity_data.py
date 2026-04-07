"""
Exporteert feature vectors en ONNX-predictions voor pariteitstest met de C# API.

Workflow:
1. Downloadt test_rows.csv, cat_codes.json, regressor.onnx en ranker.onnx uit MLflow
2. Bouwt de float[36] feature vector per rij — identiek aan hoe de C# FeatureBuilder dat doet
3. Draait ONNX inference voor regressor en ranker
4. Uploadt parity_vectors.json terug naar dezelfde MLflow run

Gebruik:
    python tests/export_parity_data.py [--mlflow-uri http://10.0.0.100:5000]
"""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

# Voeg src/ toe aan het pad zodat config importeerbaar is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CATEGORICAL, FEATURES, NUMERICAL

MLFLOW_MODEL_NAME = "rister-lgbm-regressor"
DEFAULT_MLFLOW_URI = "http://10.0.0.100:5000"


def get_latest_run_id(client: mlflow.MlflowClient, model_name: str) -> str:
    """Haal de run_id op van de laatste versie van een geregistreerd model."""
    versions = client.get_latest_versions(model_name)
    if not versions:
        raise RuntimeError(f"Geen versies gevonden voor model '{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    return latest.run_id


def download_artifacts(client: mlflow.MlflowClient, run_id: str, dest: str) -> dict:
    """Download benodigde artifacts uit een MLflow run naar een lokale map."""
    artifacts = {
        "test_rows": "test_data/test_rows.csv",
        "cat_codes": "artifacts/cat_codes.json",
        "regressor": "onnx/lgbm_regressor.onnx",
        "ranker": "onnx/lgbm_ranker.onnx",
    }
    paths = {}
    for key, artifact_path in artifacts.items():
        local = client.download_artifacts(run_id, artifact_path, dest)
        paths[key] = local
        print(f"  {artifact_path} -> {local}")
    return paths


def iso_week(dt: datetime) -> int:
    """Geeft het ISO-weeknummer terug, identiek aan C# ISOWeek.GetWeekOfYear()."""
    return dt.isocalendar()[1]


def build_feature_vector(row: pd.Series, cat_codes: dict) -> list[float]:
    """
    Bouwt een float[36] feature vector vanuit een CSV-rij, identiek aan C# FeatureBuilder.Build().

    Indices 0-10:  categorische kolommen, geencodeerd via cat_codes.json
    Indices 11-12: lat, lon (pass-through)
    Indices 13-18: sin/cos dag/maand/week (herberekend vanuit datum, zoals C# doet)
    Indices 19-35: medewerker-metrics + hoeveelheden (pass-through)
    """
    features = [0.0] * 36

    # ── Categorisch (0-10) ─────────────────────────────────────────────────────
    for i, col in enumerate(CATEGORICAL):
        val = str(row.get(col, ""))
        if col in cat_codes and val in cat_codes[col]:
            features[i] = float(cat_codes[col][val])
        else:
            features[i] = 0.0  # fallback, identiek aan C#

    # ── Numeriek (11-35) ───────────────────────────────────────────────────────
    # lat, lon
    features[11] = float(row.get("lat", 0.0))
    features[12] = float(row.get("lon", 0.0))

    # sin/cos herberekenen vanuit datum (zoals C# FeatureBuilder doet)
    date_str = str(row.get("URENVERANTW_DATUM", ""))
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        dt = datetime(2020, 1, 1)

    day = dt.weekday()  # Python: Monday=0 ... Sunday=6
    # C# DayOfWeek: Sunday=0 ... Saturday=6 → omzetten naar C# conventie
    day_csharp = (day + 1) % 7
    month = dt.month
    week = iso_week(dt)

    features[13] = math.sin(2 * math.pi * day_csharp / 7)
    features[14] = math.cos(2 * math.pi * day_csharp / 7)
    features[15] = math.sin(2 * math.pi * month / 12)
    features[16] = math.cos(2 * math.pi * month / 12)
    features[17] = math.sin(2 * math.pi * week / 52)
    features[18] = math.cos(2 * math.pi * week / 52)

    # Medewerker- en taakmetrics (pass-through, zelfde volgorde als NUMERICAL[8:])
    num_pass = [
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
    for j, col in enumerate(num_pass):
        val = row.get(col, 0.0)
        features[19 + j] = float(val) if pd.notna(val) else 0.0

    return features


def run_onnx(session: ort.InferenceSession, features_batch: np.ndarray) -> np.ndarray:
    """Draai ONNX inference op een batch feature vectors."""
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: features_batch.astype(np.float32)})
    return result[0].flatten()


def main():
    parser = argparse.ArgumentParser(description="Export parity data voor C# vergelijking")
    parser.add_argument(
        "--mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Lokaal pad voor parity_vectors.json (default: upload naar MLflow)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.MlflowClient()

    print(f"MLflow: {args.mlflow_uri}")
    run_id = get_latest_run_id(client, MLFLOW_MODEL_NAME)
    print(f"Laatste run: {run_id}")

    # Download artifacts naar tijdelijke map
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Artifacts downloaden...")
        paths = download_artifacts(client, run_id, tmpdir)

        # Laad cat_codes.json
        with open(paths["cat_codes"]) as f:
            cat_codes = json.load(f)

        # Laad test_rows.csv
        df = pd.read_csv(paths["test_rows"])
        print(f"Test data: {len(df)} rijen, {len(df.columns)} kolommen")

        # Bouw feature vectors
        all_features = []
        for _, row in df.iterrows():
            vec = build_feature_vector(row, cat_codes)
            all_features.append(vec)

        features_array = np.array(all_features, dtype=np.float32)
        print(f"Feature matrix: {features_array.shape}")

        # ONNX inference
        print("ONNX regressor laden...")
        reg_session = ort.InferenceSession(paths["regressor"])
        reg_preds = run_onnx(reg_session, features_array)

        print("ONNX ranker laden...")
        rank_session = ort.InferenceSession(paths["ranker"])
        rank_preds = run_onnx(rank_session, features_array)

        # Bouw parity output
        parity_data = []
        for i in range(len(df)):
            parity_data.append(
                {
                    "row_index": i,
                    "features": [round(float(v), 6) for v in all_features[i]],
                    "predicted_time": round(float(reg_preds[i]), 6),
                    "predicted_suitability": round(float(rank_preds[i]), 6),
                }
            )

        parity_json = json.dumps(parity_data, indent=2)

        if args.output:
            # Sla lokaal op
            output_path = Path(args.output)
            output_path.write_text(parity_json)
            print(f"Opgeslagen: {output_path}")
        else:
            # Upload naar MLflow als artifact op dezelfde run
            parity_path = os.path.join(tmpdir, "parity_vectors.json")
            with open(parity_path, "w") as f:
                f.write(parity_json)

            client.log_artifact(run_id, parity_path, artifact_path="test_data")
            print(f"Geupload naar MLflow run {run_id}: test_data/parity_vectors.json")

        # Samenvatting
        print(f"\n{'=' * 60}")
        print(f"Pariteitsdata gegenereerd voor {len(parity_data)} rijen")
        print(f"Feature vector lengte: {len(parity_data[0]['features'])}")
        print(
            f"Predicted time range: [{min(r['predicted_time'] for r in parity_data):.3f}, "
            f"{max(r['predicted_time'] for r in parity_data):.3f}]"
        )
        print(
            f"Predicted suitability range: [{min(r['predicted_suitability'] for r in parity_data):.3f}, "
            f"{max(r['predicted_suitability'] for r in parity_data):.3f}]"
        )


if __name__ == "__main__":
    main()
