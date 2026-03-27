"""
Dataprep: Samenvoegen Rister + WerkExpert
Laadt beide CSVs, brengt kolommen op één lijn via KOLOM_MAPPING,
berekent de suitability_score en slaat op als dataframe_gecombineerd.csv.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Mapping Rister-kolomnamen → WerkExpert-kolomnamen (doelkolomnamen)
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

# Kolommen alleen in Rister (geen equivalent in WerkExpert)
RISTER_ONLY = ["EquipmentGroupTypes", "planninggroupsname", "hoeveelheid_baal"]


def run():
    log.info("Start merge dataprep")

    rister_path = PROCESSED_DIR / "rister.csv"
    werkexpert_path = PROCESSED_DIR / "werkexpert.parquet"

    log.info(f"Laden {rister_path}")
    df_rister = pd.read_csv(rister_path, low_memory=False)

    log.info(f"Laden {werkexpert_path}")
    df_werkexpert = pd.read_parquet(werkexpert_path)

    # Rister: selecteer en hernoem kolommen naar doelkolomnamen
    rister_cols_beschikbaar = {k: v for k, v in KOLOM_MAPPING.items() if k in df_rister.columns}
    df_rister_sel = df_rister[
        list(rister_cols_beschikbaar.keys()) + [c for c in RISTER_ONLY if c in df_rister.columns]
    ]
    df_rister_sel = df_rister_sel.rename(columns=rister_cols_beschikbaar)
    df_rister_sel["bron"] = "rister"

    # WerkExpert: selecteer alleen de doelkolomnamen die aanwezig zijn
    doel_kolommen = list(set(KOLOM_MAPPING.values()))
    werkexpert_cols = [c for c in doel_kolommen if c in df_werkexpert.columns]
    df_werkexpert_sel = df_werkexpert[werkexpert_cols].copy()
    df_werkexpert_sel["bron"] = "werkexpert"

    # Samenvoegen
    df = pd.concat([df_rister_sel, df_werkexpert_sel], ignore_index=True)
    log.info(f"Gecombineerd: {len(df)} rijen, {df.shape[1]} kolommen")

    # Suitability score berekenen
    df["norm_ervaring"] = df.groupby("con")["med_ervaring_bewerking"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df["norm_klant"] = df.groupby("con")["med_klant_bezoeken"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df["suitability_score"] = (
        (0.6 * df["norm_ervaring"] + 0.4 * df["norm_klant"]).clip(0, 1).astype("float32")
    )
    df = df.drop(columns=["norm_ervaring", "norm_klant"])

    log.info(
        f"Suitability score — gemiddelde: {df['suitability_score'].mean():.3f}, "
        f"mediaan: {df['suitability_score'].median():.3f}"
    )

    # Opslaan
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataframe_gecombineerd.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Opgeslagen: {out_path} ({len(df)} rijen, {df.shape[1]} kolommen)")


if __name__ == "__main__":
    run()
