"""
Dataprep: WerkExpert (26 Firebird-databases)
Laadt data uit alle WerkExpert-bedrijfsdatabases, voert feature engineering uit
(inclusief geodata, TF-IDF, medewerkerprofilering) en slaat het resultaat op
als data/processed/werkexpert.csv.
"""

import logging
import os
import sys
from pathlib import Path

import firebirdsql
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BAG_COORDS_PATH,
    BODEM_GPKG_PATH,
    FIREBIRD_HOST,
    FIREBIRD_PASSWORD,
    FIREBIRD_USER,
    PROCESSED_DIR,
)

BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "rister-data")
BLOB_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATABASES = [
    "HULTER23", "HULTER24", "HULTER25",
    "KUIJPERS24", "KUIJPERS25", "KUIJPERS26",
    "MELSE09", "MELSE10", "MELSE11", "MELSE12", "MELSE13", "MELSE14",
    "MELSE15", "MELSE16", "MELSE17", "MELSE18", "MELSE19", "MELSE20",
    "MELSE21", "MELSE22", "MELSE23", "MELSE24", "MELSE25", "MELSE26",
    "POEL14", "POEL15", "POEL16", "POEL17", "POEL18", "POEL19",
    "POEL20", "POEL21", "POEL22", "POEL23", "POEL24", "POEL25", "POEL26",
    "WESTRA14", "WESTRA15", "WESTRA16", "WESTRA17", "WESTRA18", "WESTRA19",
    "WESTRA20", "WESTRA21", "WESTRA22", "WESTRA23", "WESTRA24", "WESTRA25", "WESTRA26",
    "JENNISSEN20", "JENNISSEN21", "JENNISSEN22", "JENNISSEN23", "JENNISSEN24", "JENNISSEN25", "JENNISSEN26",
    "DERKS23", "DERKS24", "DERKS25", "DERKS26",
    "DIEPEN21", "DIEPEN22", "DIEPEN23", "DIEPEN24", "DIEPEN25", "DIEPEN26",
]


def _connect(database: str):
    host, port = FIREBIRD_HOST.split(":")
    return firebirdsql.connect(
        host=host,
        port=int(port),
        database=database,
        user=FIREBIRD_USER,
        password=FIREBIRD_PASSWORD,
        charset="WIN1252",
    )


def fetch_combined(table_name: str) -> pd.DataFrame:
    """Haal tabel op uit alle databases en combineer met 'con' kolom."""
    frames = []
    for db in DATABASES:
        try:
            conn = _connect(db)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            df["con"] = db
            frames.append(df)
        except Exception as e:
            log.warning(f"{db} - {table_name} mislukt: {e}")
    if not frames:
        raise RuntimeError(f"Geen data geladen voor {table_name}")
    return pd.concat(frames, ignore_index=True)


def run():
    log.info("Start WerkExpert dataprep")

    # --- Tijdregistratie ---
    log.info("Laden TB_URENVERANTWOORDING")
    uren = fetch_combined("TB_URENVERANTWOORDING")
    uren["URENVERANTW_DATUM"] = pd.to_datetime(uren["URENVERANTW_DATUM"], errors="coerce")
    uren = uren[uren["URENVERANTW_DATUM"] >= "2000-01-01"]
    uren = uren[uren["URENVERANTW_AANTAL"] >= 0]
    uren = uren[uren["URENVERANTW_ORDDAID"] != 0]

    # --- Orders ---
    log.info("Laden TB_ORDER")
    orders = fetch_combined("TB_ORDER")
    orders["ORD_INVDATUM"] = pd.to_datetime(orders["ORD_INVDATUM"], errors="coerce")
    orders["ORD_UITVDATUM"] = pd.to_datetime(orders["ORD_UITVDATUM"], errors="coerce")
    orders = orders[orders["ORD_INVDATUM"] >= "2000-01-01"]

    # --- Tijdsoorten ---
    log.info("Laden TB_TIJDSOORT")
    tijdsoort = fetch_combined("TB_TIJDSOORT")
    werk_ids = set(tijdsoort.loc[~tijdsoort["TIJDSOORT_ONDERBREKING"].astype(bool), "TIJDSOORT_ID"])
    pauze_ids = set(tijdsoort.loc[tijdsoort["TIJDSOORT_ONDERBREKING"].astype(bool), "TIJDSOORT_ID"])

    # --- Deduplicatie & werktijd berekening ---
    key_cols = ["con", "URENVERANTW_MEDID", "URENVERANTW_ORDDAID", "URENVERANTW_DATUM"]
    werk = uren[uren["URENVERANTW_TIJDSOORT"].isin(werk_ids)].copy()
    pauze = uren[uren["URENVERANTW_TIJDSOORT"].isin(pauze_ids)].copy()

    werk_agg = werk.groupby(key_cols, as_index=False)["URENVERANTW_AANTAL"].sum().rename(
        columns={"URENVERANTW_AANTAL": "WERK_TIME"}
    )
    pauze_agg = pauze.groupby(key_cols, as_index=False)["URENVERANTW_AANTAL"].sum().rename(
        columns={"URENVERANTW_AANTAL": "PAUSE_TIME"}
    )

    df = werk_agg.merge(pauze_agg, on=key_cols, how="left")
    df["PAUSE_TIME"] = df["PAUSE_TIME"].fillna(0)
    df["REAL_WORKED_TIME"] = df["WERK_TIME"] - df["PAUSE_TIME"]
    df = df[df["REAL_WORKED_TIME"] > 0]

    p99 = df["REAL_WORKED_TIME"].quantile(0.99)
    df = df[df["REAL_WORKED_TIME"] <= p99]
    log.info(f"Na werktijd berekening: {len(df)} rijen")

    # --- Medewerker history features ---
    df = df.sort_values(key_cols).reset_index(drop=True)
    grp_emp = df.groupby(["con", "URENVERANTW_MEDID"])
    df["med_gem_tijd"] = grp_emp["REAL_WORKED_TIME"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["med_std_tijd"] = grp_emp["REAL_WORKED_TIME"].transform(
        lambda x: x.expanding().std().shift(1)
    )
    df["med_aantal_opdrachten"] = grp_emp.cumcount()

    # --- Orders koppelen ---
    log.info("Laden TB_ORDER_DA")
    orda = fetch_combined("TB_ORDER_DA")
    df = df.merge(
        orda[["con", "ORDER_DA_ID", "ORD_ID", "DIENST_ART_ID"]],
        left_on=["con", "URENVERANTW_ORDDAID"],
        right_on=["con", "ORDER_DA_ID"],
        how="left",
    )
    df = df.merge(
        orders[["con", "ORD_ID", "ORD_CODE", "ORD_OMS", "ORD_INVDATUM", "ORD_UITVDATUM",
                "ORD_SOORT", "ORD_TIJD", "PROJ_ID", "RELATIE_ID", "LOC_ID"]],
        on=["con", "ORD_ID"],
        how="left",
    )

    # --- Dienst_artikelen ---
    log.info("Laden TB_DIENST_ARTIKEL")
    dienst = fetch_combined("TB_DIENST_ARTIKEL")
    df = df.merge(
        dienst[["con", "DIENST_ART_ID", "DIENST_ART_CODE", "DIENST_ART_OMS", "DIENST_ART_TYPE"]],
        on=["con", "DIENST_ART_ID"],
        how="left",
    )

    # --- Taak gemiddelde ---
    df["taak_gem"] = df.groupby(["con", "DIENST_ART_ID"])["REAL_WORKED_TIME"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["med_snelheid_ratio"] = (df["med_gem_tijd"] / df["taak_gem"].replace(0, np.nan)).fillna(1.0)
    df["med_ervaring_bewerking"] = df.groupby(["con", "URENVERANTW_MEDID", "DIENST_ART_ID"]).cumcount()

    # --- Orderregels (hoeveelheden) ---
    log.info("Laden TB_ORDERREGEL + TB_EENHEID")
    ordrg = fetch_combined("TB_ORDERREGEL")
    eenheid = fetch_combined("TB_EENHEID")
    ordrg = ordrg.merge(eenheid[["con", "EENHEID_ID", "EENHEID_OMS"]], on=["con", "EENHEID_ID"], how="left")

    ordrg_agg = ordrg.groupby(["con", "ORDER_DA_ID"], as_index=False).agg(
        ORDRG_HOEVEELHEID=("ORDRG_HOEVEELHEID", "sum"),
        ORDRG_OMS=("ORDRG_OMS", "first"),
        EENHEID_ID=("EENHEID_ID", "first"),
        EENHEID_OMS=("EENHEID_OMS", "first"),
    )
    df = df.merge(ordrg_agg, on=["con", "ORDER_DA_ID"], how="left")

    # Eenheid categoriseren
    def categorize_unit(oms):
        if pd.isna(oms):
            return None
        oms = str(oms).lower().strip()
        if oms in ["ha", "ha."]:
            return "hectare"
        if oms in ["m3", "ltr"]:
            return "volume"
        if oms in ["ton", "kg"]:
            return "gewicht"
        if oms in ["stuk", "pak", "baal", "rol", "keer"]:
            return "stuks"
        return None

    df["eenheid_cat"] = df["EENHEID_OMS"].apply(categorize_unit)
    df["hoeveelheid_aanwezig"] = df["ORDRG_HOEVEELHEID"].notna().astype(int)

    # --- Relaties + coördinaten ---
    log.info("Laden TB_RELATIE + BAG-coördinaten")
    relatie = fetch_combined("TB_RELATIE")
    relatie = relatie.dropna(subset=["REL_POSTCODE", "REL_PLAATS"])
    relatie = relatie[relatie["LAND_ID"] == 3]
    relatie["REL_POSTCODE"] = relatie["REL_POSTCODE"].str.upper().str.replace(" ", "", regex=False)
    relatie = relatie[relatie["REL_POSTCODE"].str.match(r"^\d{4}[A-Z]{2}$", na=False)]

    bag = pd.read_csv(BAG_COORDS_PATH, low_memory=False)
    bag["postcode"] = bag["postcode"].str.upper().str.replace(" ", "", regex=False)

    # Match strategie 1: postcode + straat
    relatie = relatie.merge(
        bag[["postcode", "straat", "lat", "lon"]],
        left_on=["REL_POSTCODE", "REL_STRAAT"],
        right_on=["postcode", "straat"],
        how="left",
    )
    # Match strategie 2: alleen postcode (voor ongematchte)
    mask = relatie["lat"].isna()
    if mask.any():
        bag_pc = bag.groupby("postcode")[["lat", "lon"]].mean().reset_index()
        relatie_unmatched = relatie[mask].drop(columns=["lat", "lon", "postcode", "straat"], errors="ignore")
        relatie_unmatched = relatie_unmatched.merge(bag_pc, left_on="REL_POSTCODE", right_on="postcode", how="left")
        relatie.loc[mask, ["lat", "lon"]] = relatie_unmatched[["lat", "lon"]].values

    relatie = relatie.dropna(subset=["lat", "lon"])
    relatie = relatie.drop_duplicates(subset=["con", "RELATIE_ID"])

    df = df.merge(
        relatie[["con", "RELATIE_ID", "REL_POSTCODE", "REL_PLAATS", "lat", "lon"]],
        on=["con", "RELATIE_ID"],
        how="left",
    )

    # --- Bodemtype (geodata) ---
    if BODEM_GPKG_PATH.exists():
        log.info("Laden bodemkaart")
        bodem = gpd.read_file(BODEM_GPKG_PATH)
        df_geo = gpd.GeoDataFrame(
            df.dropna(subset=["lat", "lon"]),
            geometry=gpd.points_from_xy(df.dropna(subset=["lat", "lon"])["lon"],
                                        df.dropna(subset=["lat", "lon"])["lat"]),
            crs="EPSG:4326",
        )
        bodem = bodem.to_crs("EPSG:4326")
        df_geo = gpd.sjoin(df_geo, bodem[["Hoofdgrondsoort", "geometry"]], how="left", predicate="within")
        df.loc[df_geo.index, "Hoofdgrondsoort"] = df_geo["Hoofdgrondsoort"].values
    else:
        log.warning(f"Bodemkaart niet gevonden: {BODEM_GPKG_PATH}")
        df["Hoofdgrondsoort"] = np.nan

    # --- TF-IDF op servicebeschrijving ---
    log.info("TF-IDF + SVD op DIENST_ART_OMS")
    oms_filled = df["DIENST_ART_OMS"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)
    svd = TruncatedSVD(n_components=10, random_state=42)
    try:
        tfidf_matrix = tfidf.fit_transform(oms_filled)
        svd_result = svd.fit_transform(tfidf_matrix)
        for i in range(10):
            df[f"oms_vec_{i}"] = svd_result[:, i]
    except Exception as e:
        log.warning(f"TF-IDF mislukt: {e}")

    # --- Klantrelatie features ---
    df = df.sort_values(["con", "URENVERANTW_MEDID", "URENVERANTW_DATUM"]).reset_index(drop=True)
    df["med_klant_bezoeken"] = df.groupby(["con", "URENVERANTW_MEDID", "RELATIE_ID"]).cumcount()
    df["med_totaal_opdrachten"] = df.groupby(["con", "URENVERANTW_MEDID"]).cumcount()
    df["med_klant_ratio"] = df["med_klant_bezoeken"] / (df["med_totaal_opdrachten"] + 1)

    df["med_klant_gem_tijd"] = df.groupby(["con", "URENVERANTW_MEDID", "RELATIE_ID"])["REAL_WORKED_TIME"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["med_klant_snelheid"] = (df["med_klant_gem_tijd"] / df["med_gem_tijd"].replace(0, np.nan)).clip(0.1, 5.0)

    df["med_bewerking_gem_tijd"] = df.groupby(["con", "URENVERANTW_MEDID", "DIENST_ART_ID"])["REAL_WORKED_TIME"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["med_bewerking_snelheid"] = (df["med_bewerking_gem_tijd"] / df["med_gem_tijd"].replace(0, np.nan)).clip(0.1, 5.0)

    # --- Cyclische tijdfeatures ---
    df["dag_van_week"] = pd.to_datetime(df["URENVERANTW_DATUM"]).dt.dayofweek
    df["maand"] = pd.to_datetime(df["URENVERANTW_DATUM"]).dt.month
    df["URENVERANTW_WEEKNR"] = pd.to_datetime(df["URENVERANTW_DATUM"]).dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["URENVERANTW_WEEKNR"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["URENVERANTW_WEEKNR"] / 52)
    df["dag_sin"] = np.sin(2 * np.pi * df["dag_van_week"] / 7)
    df["dag_cos"] = np.cos(2 * np.pi * df["dag_van_week"] / 7)
    df["maand_sin"] = np.sin(2 * np.pi * df["maand"] / 12)
    df["maand_cos"] = np.cos(2 * np.pi * df["maand"] / 12)

    # --- Databases met te weinig records verwijderen ---
    counts = df.groupby("con").size()
    geldige_dbs = counts[counts >= 1000].index
    df = df[df["con"].isin(geldige_dbs)]
    log.info(f"Na DB-filter (>=1000 rijen): {len(df)} rijen, {df['con'].nunique()} databases")

    # --- Lokaal opslaan als Parquet (kleiner + sneller dan CSV) ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "werkexpert.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    log.info(f"Lokaal opgeslagen: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Uploaden naar Azure Blob Storage ---
    if BLOB_CONNECTION_STRING:
        _upload_to_blob(out_path)
    else:
        log.warning(
            "AZURE_STORAGE_CONNECTION_STRING niet ingesteld — bestand niet geupload. "
            "Voeg de connection string toe aan .env om automatisch te uploaden."
        )


def _upload_to_blob(csv_path: Path):
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        log.error("azure-storage-blob niet geïnstalleerd. Voer uit: pip install azure-storage-blob")
        return

    log.info(f"Uploaden naar Azure Blob Storage (container: {BLOB_CONTAINER})")
    client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container = client.get_container_client(BLOB_CONTAINER)

    # Container aanmaken als die nog niet bestaat
    try:
        container.create_container()
    except Exception:
        pass  # bestaat al

    blob_name = "processed/werkexpert.parquet"
    blob_client = container.get_blob_client(blob_name)
    with open(csv_path, "rb") as f:
        blob_client.upload_blob(
            f,
            overwrite=True,
            standard_blob_tier="Cool",  # goedkoper voor zelden gelezen bestanden
        )

    log.info(f"Geupload (Cool tier): {BLOB_CONTAINER}/{blob_name}")


if __name__ == "__main__":
    run()
