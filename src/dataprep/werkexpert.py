"""WerkExpert dataprep: Firebird databases → werkexpert.parquet

Draait LOKAAL (Firebird databases op Mac mini). Output handmatig uploaden naar Blob.

Bron: notebooks/2.DataPrep/WerkExpert.ipynb
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from src.config import BAG_CSV_PATH, FIREBIRD_HOST, FIREBIRD_PASSWORD, FIREBIRD_USER

log = logging.getLogger(__name__)


# ── Firebird charset monkey-patch ─────────────────────────────────────────────


def _patch_firebird_charset():
    """Monkey-patch firebirdsql om WIN1252 charset-fouten te negeren."""
    import firebirdsql.wireprotocol as wp
    from firebirdsql.consts import charset_map

    _orig = wp.WireProtocol.bytes_to_str  # noqa: F841

    def _safe_decode(self, b):
        charset = charset_map.get(self.charset, self.charset)
        return b.decode(charset, errors="replace")

    wp.WireProtocol.bytes_to_str = _safe_decode


# ── Database configuratie ─────────────────────────────────────────────────────

DATABASES = {
    # HULTER (23 t/m 25)
    "23_hulter": "/firebird/data/HULTER_23 - kopie.GDB",
    "24_hulter": "/firebird/data/HULTER_24 - kopie.GDB",
    "25_hulter": "/firebird/data/HULTER_25 - kopie.GDB",
    # KUIJPERS (24 t/m 26)
    "24_kuijpers": "/firebird/data/HOOFD_24.GDB",
    "25_kuijpers": "/firebird/data/HOOFD_25.GDB",
    "26_kuijpers": "/firebird/data/HOOFD_26.GDB",
    # MELSE (09 t/m 25)
    "09_melse": "/firebird/data/MELSE2009.GDB",
    **{f"{jaar}_melse": f"/firebird/data/MELSE20{jaar}.GDB" for jaar in range(10, 26)},
    # POEL (14 t/m 25)
    **{f"{jaar}_poel": f"/firebird/data/POEL_20{jaar}.GDB" for jaar in range(14, 26)},
    # WESTRA (14 t/m 25)
    **{f"{jaar}_westra": f"/firebird/data/WESTRA_{jaar}.GDB" for jaar in range(14, 26)},
    # JENNISSEN (20 t/m 25)
    **{f"{jaar}_jennissen": f"/firebird/data/jennissen_{jaar}.GDB" for jaar in range(20, 26)},
    # DERKS (23 t/m 25)
    "23_derks": "/firebird/data/DERKS_23.GDB",
    "24_derks": "/firebird/data/DERKS_24.GDB",
    "25_derks": "/firebird/data/DERKS_25.GDB",
    # DIEPEN (21 t/m 25)
    **{f"{jaar}_diepen": f"/firebird/data/DIEPEN_{jaar}.GDB" for jaar in range(21, 26)},
}


def open_connections(host: str) -> dict:
    """Open Firebird connecties naar alle databases."""
    import firebirdsql

    host_name, port = host.split(":") if ":" in host else (host, 3050)
    connections = {}
    for naam, pad in DATABASES.items():
        try:
            connections[naam] = firebirdsql.connect(
                host=host_name,
                port=int(port),
                database=pad,
                user=FIREBIRD_USER,
                password=FIREBIRD_PASSWORD,
                charset="WIN1252",
            )
        except Exception as e:
            log.warning("Kan %s niet openen: %s", naam, e)
    log.info("%d/%d databases verbonden", len(connections), len(DATABASES))
    return connections


def fetch_combined(connections: dict, table_name: str, columns: str = "*") -> pd.DataFrame:
    """Haal data op uit alle databases en combineer in één DataFrame."""
    frames = []
    for db_name, conn in connections.items():
        try:
            df = pd.read_sql(f"SELECT {columns} FROM {table_name}", conn)
            df["con"] = db_name
            frames.append(df)
        except Exception as e:
            log.warning("Fout bij %s.%s: %s", db_name, table_name, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


# ── Data extractie ────────────────────────────────────────────────────────────


def extract_tables(connections: dict) -> dict[str, pd.DataFrame]:
    """Haal alle benodigde tabellen op uit Firebird."""
    tables = {}

    tables["urenverantw"] = fetch_combined(connections, "TB_URENVERANTWOORDING")
    tables["tijdsoort"] = fetch_combined(connections, "TB_TIJDSOORT")
    tables["order_da"] = fetch_combined(connections, "TB_ORDER_DA")
    tables["dienst_artikel"] = fetch_combined(connections, "TB_DIENST_ARTIKEL")
    tables["orderregel"] = fetch_combined(connections, "TB_ORDERREGEL")
    tables["eenheid"] = fetch_combined(connections, "TB_EENHEID")
    tables["order"] = fetch_combined(connections, "TB_ORDER")
    tables["relatie"] = fetch_combined(
        connections,
        "TB_RELATIE",
        columns="REL_HUISNUMMER, REL_STRAAT, REL_POSTCODE, RELATIE_ID, LAND_ID, REL_PLAATS",
    )
    tables["machine"] = fetch_combined(connections, "TB_MACHINE")

    for name, df in tables.items():
        log.info("Tabel %s: %d rijen", name, len(df))

    return tables


# ── Tijd-aggregatie ───────────────────────────────────────────────────────────


def aggregate_work_time(tables: dict) -> pd.DataFrame:
    """Bereken netto werktijd: werk_tijd - pauze_tijd per medewerker/opdracht/dag."""
    urenverantw = tables["urenverantw"].copy()
    tijdsoort = tables["tijdsoort"].copy()

    # Merge tijdsoort info
    urenverantw = urenverantw.merge(
        tijdsoort[["TIJDSOORT_ID", "TIJDSOORT_ONDERBREKING", "con"]],
        on=["TIJDSOORT_ID", "con"],
        how="left",
    )

    # Datums opschonen
    urenverantw["URENVERANTW_DATUM"] = pd.to_datetime(
        urenverantw["URENVERANTW_DATUM"], errors="coerce"
    )
    urenverantw = urenverantw[urenverantw["URENVERANTW_DATUM"] >= "2000-01-01"]
    urenverantw = urenverantw[urenverantw["URENVERANTW_AANTAL"] >= 0]
    urenverantw = urenverantw[urenverantw["URENVERANTW_ORDDAID"] != 0]

    group_cols = ["con", "URENVERANTW_MEDID", "URENVERANTW_ORDDAID", "URENVERANTW_DATUM"]

    # Werkuren: tijdsoort 1 of 3 (werk)
    werk = urenverantw[urenverantw["URENVERANTW_TIJDSOORT"].isin([1, 3])]

    # Deduplicatie: sommeer uren voor dezelfde medewerker/opdracht/dag
    werk_agg = werk.groupby(group_cols)["URENVERANTW_AANTAL"].sum().reset_index()
    werk_agg = werk_agg.rename(columns={"URENVERANTW_AANTAL": "WERK_TIME"})

    # Pauzes: niet tijdsoort 1/3 EN is een onderbreking
    pauzes = urenverantw[
        (~urenverantw["URENVERANTW_TIJDSOORT"].isin([3, 1]))
        & (urenverantw["TIJDSOORT_ONDERBREKING"] == True)  # noqa: E712
    ]
    pauze_agg = pauzes.groupby(group_cols)["URENVERANTW_AANTAL"].sum().reset_index()
    pauze_agg = pauze_agg.rename(columns={"URENVERANTW_AANTAL": "PAUSE_TIME"})

    # Netto werktijd
    result = werk_agg.merge(pauze_agg, on=group_cols, how="left")
    result["PAUSE_TIME"] = result["PAUSE_TIME"].fillna(0)
    result["REAL_WORKED_TIME"] = result["WERK_TIME"] - result["PAUSE_TIME"]

    # Outlier removal
    p99 = result["REAL_WORKED_TIME"].quantile(0.99)
    result = result[(result["REAL_WORKED_TIME"] > 0) & (result["REAL_WORKED_TIME"] <= p99)]

    log.info("Werktijd geaggregeerd: %d rijen", len(result))
    return result


# ── Merge met dimensie-tabellen ───────────────────────────────────────────────


def merge_dimensions(work_time: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Merge werktijd met order, dienst, machine, orderregel en eenheid."""
    df = work_time.copy()

    # Order details via ORDER_DA
    order_da = tables["order_da"]
    df = df.merge(
        order_da,
        left_on=["con", "URENVERANTW_ORDDAID"],
        right_on=["con", "ORDER_DA_ID"],
        how="left",
    )

    # Dienst/artikel info
    dienst = tables["dienst_artikel"]
    df = df.merge(dienst, on=["con", "DIENST_ART_ID"], how="left")

    # Orderregel + eenheid
    orderregel = tables["orderregel"]
    eenheid = tables["eenheid"]
    orderregel = orderregel.merge(eenheid, on=["con", "EENHEID_ID"], how="left")
    df = df.merge(orderregel, on=["con", "ORDER_DA_ID"], how="left")

    # Order
    order = tables["order"]
    df = df.merge(order, left_on=["con", "ORDER_ID"], right_on=["con", "ORD_ID"], how="left")

    # Machine
    machine = tables["machine"]
    df = df.merge(machine, on=["con", "DIENST_ART_ID"], how="left")

    log.info("Na dimensie-merge: %s", df.shape)
    return df


# ── BAG geocoding ─────────────────────────────────────────────────────────────


def geocode_relations(tables: dict, bag_csv_path: str) -> pd.DataFrame:
    """Geocode klant-adressen via BAG postcode lookup."""
    relatie = tables["relatie"].copy()
    relatie = relatie.dropna(subset=["REL_HUISNUMMER", "REL_STRAAT", "REL_POSTCODE"])
    relatie = relatie[relatie["LAND_ID"] == 3]  # Alleen Nederland

    # Normaliseer
    relatie["REL_POSTCODE"] = relatie["REL_POSTCODE"].str.replace(" ", "", regex=False).str.upper()
    relatie["REL_STRAAT"] = relatie["REL_STRAAT"].str.upper().str.replace(" ", "", regex=False)
    relatie = relatie[~relatie["REL_STRAAT"].isin(["POSBUS", "POSTBUS", "P.O.BOX", ""])]

    bag = pd.read_csv(bag_csv_path)
    bag_pc = bag.drop_duplicates(subset=["pc"])

    # Strategie 1: match op postcode
    valid_pc = relatie[relatie["REL_POSTCODE"].str.contains(r"^\d{4}[A-Z]{2}$", na=False)]
    matched = valid_pc.merge(bag_pc, left_on="REL_POSTCODE", right_on="pc", how="left")
    found = matched[matched["lat"].notna()]
    not_found = matched[matched["lat"].isna()].drop(columns=["lat", "lon", "pc"], errors="ignore")

    # Strategie 2: match op straatnaam
    invalid_pc = relatie[~relatie["REL_POSTCODE"].str.contains(r"^\d{4}[A-Z]{2}$", na=False)]
    remaining = pd.concat([not_found, invalid_pc], ignore_index=True)

    bag_straat = bag.drop_duplicates(subset=["straat", "hnr"])
    bag_straat["straat"] = bag_straat["straat"].str.upper().str.replace(" ", "", regex=False)
    matched2 = remaining.merge(bag_straat, left_on="REL_STRAAT", right_on="straat", how="left")
    found2 = matched2[matched2["lat"].notna()]
    not_found2 = matched2[matched2["lat"].isna()].drop(
        columns=["lat", "lon", "straat", "hnr", "pc"], errors="ignore"
    )

    # Strategie 3: match op eerste 4 cijfers postcode
    bag_4 = bag_pc.copy()
    bag_4["pc4"] = bag_4["pc"].str[:4]
    bag_4 = bag_4.drop_duplicates(subset=["pc4"])
    not_found2["pc4"] = not_found2["REL_POSTCODE"].str[:4]
    matched3 = not_found2.merge(bag_4[["pc4", "lat", "lon"]], on="pc4", how="left")
    found3 = matched3[matched3["lat"].notna()]

    # Combineer
    df_rel = pd.concat([found, found2, found3], ignore_index=True)
    df_rel = df_rel.drop_duplicates(subset=["con", "RELATIE_ID"])
    df_rel = df_rel.dropna(subset=["lat", "lon"])

    log.info("Geocoded relaties: %d", len(df_rel))
    return df_rel[["con", "RELATIE_ID", "REL_POSTCODE", "lat", "lon"]]


# ── Feature engineering ───────────────────────────────────────────────────────


def add_employee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding window en cumulatieve medewerker-features."""
    df = df.sort_values("URENVERANTW_DATUM")

    grouped = df.groupby(["con", "URENVERANTW_MEDID"])["REAL_WORKED_TIME"]
    df["med_gem_tijd"] = grouped.expanding().mean().shift(1).reset_index(level=[0, 1], drop=True)
    df["med_std_tijd"] = grouped.expanding().std().shift(1).reset_index(level=[0, 1], drop=True)
    df["med_aantal_opdrachten"] = (
        grouped.expanding().count().shift(1).reset_index(level=[0, 1], drop=True)
    )

    # Taak-gemiddelde
    df["taak_gem"] = (
        df.groupby(["con", "DIENST_ART_ID"])["REAL_WORKED_TIME"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )

    # Ervaring per bewerking
    df["med_ervaring_bewerking"] = df.groupby(
        ["con", "URENVERANTW_MEDID", "DIENST_ART_ID"]
    ).cumcount()

    # Klant-bezoeken en totaal opdrachten
    df["med_klant_bezoeken"] = df.groupby(["con", "URENVERANTW_MEDID", "RELATIE_ID"]).cumcount()
    df["med_totaal_opdrachten"] = df.groupby(["con", "URENVERANTW_MEDID"]).cumcount()

    # Ratio's
    df["med_klant_ratio"] = df["med_klant_bezoeken"] / (df["med_totaal_opdrachten"] + 1)

    # Klant-specifieke snelheid
    df["med_klant_gem_tijd"] = (
        df.groupby(["con", "URENVERANTW_MEDID", "RELATIE_ID"])["REAL_WORKED_TIME"]
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
        df.groupby(["con", "URENVERANTW_MEDID", "DIENST_ART_ID"])["REAL_WORKED_TIME"]
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
    df["URENVERANTW_DATUM"] = pd.to_datetime(df["URENVERANTW_DATUM"])
    dag = df["URENVERANTW_DATUM"].dt.dayofweek
    maand = df["URENVERANTW_DATUM"].dt.month
    week = df["URENVERANTW_DATUM"].dt.isocalendar().week.astype(int)

    df["dag_sin"] = np.sin(2 * np.pi * dag / 7)
    df["dag_cos"] = np.cos(2 * np.pi * dag / 7)
    df["maand_sin"] = np.sin(2 * np.pi * maand / 12)
    df["maand_cos"] = np.cos(2 * np.pi * maand / 12)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)
    return df


def add_quantity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categoriseer hoeveelheden per eenheidstype."""
    if "EENHEID_OMS" not in df.columns or "ORDRG_HOEVEELHEID" not in df.columns:
        df["hoeveelheid_volume"] = np.nan
        df["hoeveelheid_gewicht"] = np.nan
        df["hoeveelheid_stuks"] = np.nan
        df["hoeveelheid_aanwezig"] = 0
        return df

    oms = df["EENHEID_OMS"].str.lower().str.strip()
    hv = df["ORDRG_HOEVEELHEID"]

    df["hoeveelheid_volume"] = hv.where(oms.isin(["m3", "m3.", "ltr", "ltr."]))
    df["hoeveelheid_gewicht"] = hv.where(oms.isin(["ton", "kg"]))
    df["hoeveelheid_stuks"] = hv.where(
        oms.isin(["stuk", "st.", "st", "pak", "baal", "rol", "keer", "kr."])
    )
    df["hoeveelheid_aanwezig"] = (
        df[["hoeveelheid_volume", "hoeveelheid_gewicht", "hoeveelheid_stuks"]]
        .notna()
        .any(axis=1)
        .astype(int)
    )
    return df


def add_tfidf_features(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """TF-IDF + SVD op dienst-omschrijvingen."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    dienst_oms = df["DIENST_ART_OMS"].fillna("onbekend").astype(str)

    tfidf = TfidfVectorizer(
        max_features=500, ngram_range=(1, 2), min_df=2, analyzer="word", lowercase=True
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    tfidf_matrix = tfidf.fit_transform(dienst_oms)
    vectors = svd.fit_transform(tfidf_matrix)

    for i in range(n_components):
        df[f"oms_vec_{i}"] = vectors[:, i]

    return df


# ── Main ─────────────────────────────────────────────────────────────────────


def main(firebird_host: str, output_dir: str) -> str:
    """Voer de volledige WerkExpert dataprep pipeline uit."""
    os.makedirs(output_dir, exist_ok=True)

    _patch_firebird_charset()
    connections = open_connections(firebird_host)

    try:
        # Extract
        tables = extract_tables(connections)

        # Werktijd aggregatie
        work_time = aggregate_work_time(tables)

        # Merge dimensie-tabellen
        df = merge_dimensions(work_time, tables)

        # Geocoding
        bag_csv = str(BAG_CSV_PATH)
        geo = geocode_relations(tables, bag_csv)
        df = df.merge(geo, on=["con", "RELATIE_ID"], how="left")

        # Feature engineering
        df = add_employee_features(df)
        df = add_temporal_features(df)
        df = add_quantity_features(df)
        df = add_tfidf_features(df)

        # Filter: minimaal 1000 rijen per database
        counts = df["con"].value_counts()
        df = df[df["con"].isin(counts[counts >= 1000].index)]

        # Filter: rijen zonder coordinaten verwijderen
        df = df.dropna(subset=["lat", "lon"])

        log.info("Finale dataset: %s", df.shape)

        # Opslaan als parquet
        output_path = os.path.join(output_dir, "werkexpert.parquet")
        df.to_parquet(output_path, index=False, compression="snappy")
        log.info("Opgeslagen: %s", output_path)

        # Ook CSV voor backwards compatibility
        csv_path = os.path.join(output_dir, "werkexpert.csv")
        df.to_csv(csv_path, index=False)
        log.info("Opgeslagen: %s", csv_path)

        return output_path
    finally:
        for conn in connections.values():
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(
        description="WerkExpert dataprep: Firebird → werkexpert.parquet"
    )
    parser.add_argument(
        "--firebird-host",
        default=os.environ.get("FIREBIRD_HOST", FIREBIRD_HOST),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "data"),
    )
    args = parser.parse_args()
    main(args.firebird_host, args.output_dir)
