"""
Dataprep: Rister (PostgreSQL)
Laadt data uit de Rister-database, voert feature engineering uit
en slaat het resultaat op als data/processed/rister.csv.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BAG_COORDS_PATH, POSTGRES_URI, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_table(engine, schema, table, columns=None):
    """Laad een tabel uit de database; hernoem ID/Name kolommen automatisch."""
    col_sql = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    df = pd.read_sql(f'SELECT {col_sql} FROM "{schema}"."{table}"', engine)
    rename = {}
    for col in df.columns:
        if col.endswith("Id") and col != f"{table}Id":
            pass
        if col == "Id":
            rename["Id"] = f"{table}Id"
        if col == "Name":
            rename["Name"] = f"{table}Name"
    return df.rename(columns=rename)


def run():
    log.info("Start Rister dataprep")
    engine = create_engine(POSTGRES_URI)

    # --- Orders ---
    orders = get_table(engine, "public", "Orders")
    log.info(f"Orders: {len(orders)} rijen")

    # --- OrderServiceLine ---
    osl = get_table(engine, "public", "OrderServiceLine")

    # --- OrderServices ---
    os_ = get_table(engine, "public", "OrderServices")
    os_ = os_.merge(osl, on="OrderServiceLineId", how="left", suffixes=("", "_osl"))

    # --- ServiceLineUnits ---
    slu = get_table(engine, "public", "ServiceLineUnits")

    # --- ServiceLineEquipment ---
    sle = get_table(engine, "public", "ServiceLineEquipment")

    # --- Relations ---
    relations = get_table(engine, "public", "Relations")

    # --- Addresses + BAG coördinaten ---
    addresses = get_table(engine, "public", "Addresses")
    bag = pd.read_csv(BAG_COORDS_PATH, low_memory=False)
    addresses["PostalCode"] = addresses["PostalCode"].str.upper().str.replace(" ", "", regex=False)
    bag["postcode"] = bag["postcode"].str.upper().str.replace(" ", "", regex=False)
    addresses = addresses.merge(
        bag[["postcode", "lat", "lon"]], left_on="PostalCode", right_on="postcode", how="left"
    )

    # --- Services & ServiceLines ---
    services = get_table(engine, "public", "Services")
    service_lines = get_table(engine, "public", "ServiceLines")

    # --- Equipment ---
    equipment = get_table(engine, "public", "Equipment")
    eteg = get_table(engine, "public", "EquipmentToEquipmentGroups")
    eg = get_table(engine, "public", "EquipmentGroups")
    equipment = equipment.merge(eteg, on="EquipmentId", how="left").merge(
        eg, on="EquipmentGroupId", how="left", suffixes=("", "_eg")
    )

    # --- PlanningGroups ---
    pg = get_table(engine, "public", "PlanningGroups")

    # --- TimeRegistration ---
    time_reg = get_table(engine, "public", "TimeRegistration")

    # --- Samenvoegen ---
    df = time_reg.copy()
    df = df.merge(os_, on="OrderServiceLineId", how="left")
    df = df.merge(orders, on="OrderId", how="left", suffixes=("", "_order"))
    df = df.merge(slu, on="ServiceLineUnitId", how="left")
    df = df.merge(sle, on="ServiceLineEquipmentId", how="left")
    df = df.merge(relations, on="ClientId", how="left", suffixes=("", "_rel"))
    df = df.merge(addresses, on="AddressId", how="left", suffixes=("", "_addr"))
    df = df.merge(services, on="ServiceId", how="left", suffixes=("", "_svc"))
    df = df.merge(service_lines, on="ServiceLineId", how="left", suffixes=("", "_sl"))
    df = df.merge(equipment, on="EquipmentId", how="left", suffixes=("", "_eq"))
    df = df.merge(pg, on="PlanningGroupId", how="left", suffixes=("", "_pg"))

    log.info(f"Na samenvoegen: {len(df)} rijen, {df.shape[1]} kolommen")

    # --- Filtering ---
    df = df[df["OrderStatus"].isin([5, 7, 8, 9, 10])]
    df = df.dropna(subset=["TimeRegistrationId"])
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True, errors="coerce")
    df["Datum"] = df["StartTime"].dt.date
    df = df.sort_values(["EmployeeId", "StartTime"]).reset_index(drop=True)

    log.info(f"Na statusfilter: {len(df)} rijen")

    # --- Employee history features (expanding, shifted) ---
    grp_emp = df.groupby("EmployeeId")
    df["med_gem_tijd"] = grp_emp["hoeveelheid_uur"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["med_std_tijd"] = grp_emp["hoeveelheid_uur"].transform(
        lambda x: x.expanding().std().shift(1)
    )
    df["med_aantal_opdrachten"] = grp_emp.cumcount()

    df["taak_gem"] = df.groupby("ServiceLineId")["hoeveelheid_uur"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    df["med_ervaring_bewerking"] = df.groupby(["EmployeeId", "ServiceId"]).cumcount()
    df["med_klant_bezoeken"] = df.groupby(["EmployeeId", "ClientId"]).cumcount()
    df["med_totaal_opdrachten"] = df.groupby("EmployeeId").cumcount()
    df["med_klant_ratio"] = df["med_klant_bezoeken"] / (df["med_totaal_opdrachten"] + 1)

    df["med_klant_gem_tijd"] = df.groupby(["EmployeeId", "ClientId"])["hoeveelheid_uur"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["med_klant_snelheid"] = (
        df["med_klant_gem_tijd"] / df["med_gem_tijd"].replace(0, np.nan)
    ).clip(0.1, 5.0)

    df["med_bewerking_gem_tijd"] = df.groupby(["EmployeeId", "ServiceId"])[
        "hoeveelheid_uur"
    ].transform(lambda x: x.expanding().mean().shift(1))
    df["med_bewerking_snelheid"] = (
        df["med_bewerking_gem_tijd"] / df["med_gem_tijd"].replace(0, np.nan)
    ).clip(0.1, 5.0)

    # --- Datum/tijd features (cyclisch) ---
    df["dag_van_week"] = df["StartTime"].dt.dayofweek
    df["maand"] = df["StartTime"].dt.month
    df["week"] = df["StartTime"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df["dag_sin"] = np.sin(2 * np.pi * df["dag_van_week"] / 7)
    df["dag_cos"] = np.cos(2 * np.pi * df["dag_van_week"] / 7)
    df["maand_sin"] = np.sin(2 * np.pi * df["maand"] / 12)
    df["maand_cos"] = np.cos(2 * np.pi * df["maand"] / 12)

    # --- Units pivot ---
    unit_cols = [
        "hoeveelheid_uur",
        "hoeveelheid_baal",
        "hoeveelheid_big_bag",
        "hoeveelheid_dag",
        "hoeveelheid_hectare",
        "hoeveelheid_kubieke_meter",
        "hoeveelheid_meter",
        "hoeveelheid_milimeter",
        "hoeveelheid_stuk",
        "hoeveelheid_ton",
    ]
    for col in unit_cols:
        if col not in df.columns:
            df[col] = 0.0

    df["heeft_extra_eenheid"] = df[[c for c in unit_cols if c != "hoeveelheid_uur"]].sum(axis=1) > 0

    # --- Outlier filtering ---
    df = df[df["hoeveelheid_uur"] > 0]
    p99 = df["hoeveelheid_uur"].quantile(0.99)
    df = df[df["hoeveelheid_uur"] <= p99]

    # --- Eindfilter: alleen records met extra eenheden ---
    df = df[df["heeft_extra_eenheid"]]

    log.info(f"Na outlier + eenheidsfilter: {len(df)} rijen")

    # --- Opslaan ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "rister.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Opgeslagen: {out_path}")


if __name__ == "__main__":
    run()
