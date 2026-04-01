"""Rister dataprep: PostgreSQL → rister.csv

Bron: notebooks/2.DataPrep/rister.ipynb
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.config import BAG_CSV_PATH

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_table(engine, schema: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Lees een tabel uit PostgreSQL en hernoem Id/Name kolommen."""
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    df = pd.read_sql(f'SELECT {cols} FROM "{schema}"."{table}"', engine)
    df = df.rename(columns={"Id": f"{table.lower()}_id", "Name": f"{table.lower()}name"})
    return df


# ── Extract & merge ──────────────────────────────────────────────────────────


def extract_and_merge(engine, bag_csv_path: str) -> pd.DataFrame:
    """Haal alle tabellen op en merge tot één DataFrame."""
    gt = lambda schema, table: get_table(engine, schema, table)  # noqa: E731

    # Basis: Orders + ServiceLine + Services
    orders = gt("Orders", "Orders")
    order_service_line = gt("Orders", "OrderServiceLine")
    order_services = gt("Orders", "OrderServices")
    order_services = order_services.merge(
        order_service_line, left_on="orderservices_id", right_on="OrderServiceId", how="right"
    )
    df = order_services.merge(orders, left_on="OrderId", right_on="orders_id", how="left")

    # Units
    service_line_units = gt("Orders", "ServiceLineUnits")
    units = gt("Shared", "Units")
    service_line_units = service_line_units.merge(units, left_on="UnitId", right_on="units_id")
    df = df.merge(
        service_line_units, left_on="orderserviceline_id", right_on="ServiceLineId", how="left"
    )

    # Equipment op service line
    service_line_equipment = gt("Orders", "ServiceLineEquipment")
    df = df.merge(
        service_line_equipment, left_on="orderserviceline_id", right_on="ServiceLineId", how="left"
    )

    # Relatie (klant)
    relation = gt("Management", "Relations").drop(columns="TenantId")
    df = df.merge(relation, left_on="TenantId", right_on="relations_id", how="left")

    # Adressen + BAG geocoding
    addresses = gt("Management", "Addresses")
    addresses["PostalCode"] = addresses["PostalCode"].str.upper().str.replace(" ", "", regex=False)
    bag = pd.read_csv(bag_csv_path).drop_duplicates(subset=["pc"])
    addresses = addresses.merge(bag, left_on="PostalCode", right_on="pc", how="left")
    df = df.merge(addresses, left_on="PostalAddressId", right_on="addresses_id", how="left")

    # Services + ServiceLines
    services = gt("Management", "Services")
    df = df.merge(
        services,
        left_on=["TenantId", "ServiceId"],
        right_on=["RelationId", "services_id"],
        how="left",
    )
    service_lines = gt("Management", "ServiceLines")
    df = df.merge(service_lines, left_on="ServiceLineId_x", right_on="servicelines_id", how="left")

    # Equipment + EquipmentGroups
    equipment = gt("Management", "Equipment")
    eq_to_groups = gt("Management", "EquipmentToEquipmentGroups")
    eq_groups = gt("Management", "EquipmentGroups").drop(columns="RelationId")
    equipment = equipment.merge(
        eq_to_groups, left_on="equipment_id", right_on="EquipmentId", how="left"
    )
    equipment = equipment.merge(
        eq_groups, left_on="EquipmentGroupId", right_on="equipmentgroups_id", how="left"
    )
    df = df.merge(equipment, left_on="EquipmentId", right_on="equipment_id", how="left")

    # PlanningGroups
    planning_groups = gt("Management", "PlanningGroups")
    df = df.merge(
        planning_groups, left_on="PlanningGroupId", right_on="planninggroups_id", how="left"
    )

    # TimeRegistration
    time_reg = gt("Orders", "TimeRegistration")
    df = df.merge(time_reg, on=["OrderId", "EmployeeId"], how="left")

    log.info("Tabellen gemerged: %s", df.shape)
    return df


# ── Filters ──────────────────────────────────────────────────────────────────


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter op status en aanwezigheid tijdregistratie."""
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True, errors="coerce")
    df["Datum"] = df["StartTime"].dt.date

    df = df[df["OrderStatus"].isin([5, 7, 8, 9, 10])]
    df = df[df["timeregistration_id"].notna()]
    log.info("Na filtering: %s", df.shape)
    return df


# ── Feature engineering ──────────────────────────────────────────────────────


def add_employee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding window features per medewerker."""
    grouped = df.groupby("EmployeeId")["ManHourDuration"]

    df["med_gem_tijd"] = grouped.expanding().mean().shift(1).reset_index(level=[0, 1], drop=True)
    df["med_std_tijd"] = grouped.expanding().std().shift(1).reset_index(level=[0, 1], drop=True)
    df["med_aantal_opdrachten"] = (
        grouped.expanding().count().shift(1).reset_index(level=[0, 1], drop=True)
    )

    # Taak-gemiddelde
    df["taak_gem"] = (
        df.groupby("ServiceId_x")["ManHourDuration"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )

    # Ervaring per bewerking
    df["med_ervaring_bewerking"] = df.groupby(["EmployeeId", "ServiceId_x"]).cumcount()

    # Sorteer op datum voor cumulatieve features
    df = df.sort_values("Datum")

    # Klant-bezoeken
    df["med_klant_bezoeken"] = df.groupby(["EmployeeId", "ClientId"]).cumcount()
    df["med_totaal_opdrachten"] = df.groupby("EmployeeId").cumcount()
    df["med_klant_ratio"] = df["med_klant_bezoeken"] / (df["med_totaal_opdrachten"] + 1)

    # Klant-specifieke snelheid
    df["med_klant_gem_tijd"] = (
        df.groupby(["EmployeeId", "ClientId"])["ManHourDuration"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1, 2], drop=True)
    )
    df["med_klant_snelheid"] = (
        (df["med_klant_gem_tijd"] / df["med_gem_tijd"].replace(0, np.nan))
        .fillna(1.0)
        .clip(0.1, 5.0)
    )

    # Bewerking-specifieke snelheid
    df["med_bewerking_gem_tijd"] = (
        df.groupby(["EmployeeId", "ServiceId_x"])["ManHourDuration"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1, 2], drop=True)
    )
    df["med_bewerking_snelheid"] = (
        (df["med_bewerking_gem_tijd"] / df["med_gem_tijd"].replace(0, np.nan))
        .fillna(1.0)
        .clip(0.1, 5.0)
    )

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclische sin/cos encodering voor dag, week, maand."""
    df["Datum"] = pd.to_datetime(df["Datum"])
    dag = df["Datum"].dt.dayofweek
    maand = df["Datum"].dt.month
    week = df["Datum"].dt.isocalendar().week.astype(int)

    df["dag_sin"] = np.sin(2 * np.pi * dag / 7)
    df["dag_cos"] = np.cos(2 * np.pi * dag / 7)
    df["maand_sin"] = np.sin(2 * np.pi * maand / 12)
    df["maand_cos"] = np.cos(2 * np.pi * maand / 12)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)
    return df


def pivot_units(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot unit-hoeveelheden naar kolommen; houd alleen 'Uur' rijen."""
    pivot = df.groupby(["orderserviceline_id", "unitsname"])["Amount"].sum().unstack(fill_value=0)
    pivot.columns = [f"hoeveelheid_{col.lower().replace(' ', '_')}" for col in pivot.columns]
    pivot = pivot.reset_index()

    df_uur = df[df["unitsname"] == "Uur"].drop_duplicates("orderserviceline_id")
    df = df_uur.merge(pivot, on="orderserviceline_id", how="inner")

    # Controleer op extra eenheden
    hoeveelheid_cols = [
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
    available = [c for c in hoeveelheid_cols if c in df.columns]
    df["heeft_extra_eenheid"] = df[available].fillna(0).sum(axis=1) > 0 if available else False

    return df


# ── Kolom selectie & output ──────────────────────────────────────────────────

BEST_COLUMNS = [
    "EmployeeId",
    "ServiceLineId_x",
    "PlanningGroupId",
    "EquipmentGroupId",
    "ClientId",
    "TenantId_x",
    "orderserviceline_id",
    "servicelinesname",
    "equipmentname",
    "EquipmentType_y",
    "EquipmentGroupTypes",
    "planninggroupsname",
    "PostalCode",
    "lat",
    "lon",
    "StartTime",
    "Datum",
    "dag_sin",
    "dag_cos",
    "maand_sin",
    "maand_cos",
    "week_sin",
    "week_cos",
    "med_gem_tijd",
    "med_std_tijd",
    "med_aantal_opdrachten",
    "med_ervaring_bewerking",
    "taak_gem",
    "med_klant_bezoeken",
    "med_klant_ratio",
    "med_klant_snelheid",
    "med_bewerking_snelheid",
    "med_klant_gem_tijd",
    "med_bewerking_gem_tijd",
    "med_totaal_opdrachten",
    "hoeveelheid_baal",
    "hoeveelheid_big_bag",
    "hoeveelheid_dag",
    "hoeveelheid_hectare",
    "hoeveelheid_kubieke_meter",
    "hoeveelheid_meter",
    "hoeveelheid_milimeter",
    "hoeveelheid_stuk",
    "hoeveelheid_ton",
    "heeft_extra_eenheid",
    "hoeveelheid_uur",
]


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Selecteer kolommen, verwijder outliers, filter op extra eenheden."""
    cols = [c for c in BEST_COLUMNS if c in df.columns]
    df = df[cols]

    # Verwijder nul/negatieve uren en outliers
    df = df[df["hoeveelheid_uur"] > 0]
    grens = df["hoeveelheid_uur"].quantile(0.99)
    df = df[df["hoeveelheid_uur"] <= grens]

    # Alleen rijen met extra eenheden
    df = df[df["heeft_extra_eenheid"]]

    log.info("Na selectie en cleaning: %s", df.shape)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────


def main(postgres_uri: str, output_dir: str) -> str:
    """Voer de volledige Rister dataprep pipeline uit."""
    os.makedirs(output_dir, exist_ok=True)
    engine = create_engine(postgres_uri)

    bag_csv = str(BAG_CSV_PATH)

    df = extract_and_merge(engine, bag_csv)
    df = apply_filters(df)
    df = add_employee_features(df)
    df = add_temporal_features(df)
    df = pivot_units(df)
    df = select_and_clean(df)

    output_path = os.path.join(output_dir, "rister.csv")
    df.to_csv(output_path, index=False)
    log.info("Opgeslagen: %s (%d rijen)", output_path, len(df))
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Rister dataprep: PostgreSQL → rister.csv")
    parser.add_argument(
        "--postgres-uri",
        default=os.environ.get(
            "POSTGRES_URI",
            "postgresql://samberkhout@localhost:5432/rister_prod17",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "data"),
    )
    args = parser.parse_args()
    main(args.postgres_uri, args.output_dir)
