"""Merge Rister + WerkExpert datasets en bereken suitability score.

Bron: notebooks/2.DataPrep/kolommen_samenvoegen.ipynb
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import KOLOM_MAPPING, RISTER_ONLY

log = logging.getLogger(__name__)


def load_rister(path: str) -> pd.DataFrame:
    """Laad Rister CSV, filter kolommen en hernoem naar WerkExpert-namen."""
    df = pd.read_csv(path, low_memory=False)

    rister_cols = list(KOLOM_MAPPING.keys()) + RISTER_ONLY
    beschikbaar = [c for c in rister_cols if c in df.columns]
    df = df[beschikbaar]

    rename_map = {k: v for k, v in KOLOM_MAPPING.items() if k in beschikbaar}
    df = df.rename(columns=rename_map)
    df["bron"] = "rister"

    log.info("Rister geladen: %s", df.shape)
    return df


def load_werkexpert(path: str) -> pd.DataFrame:
    """Laad WerkExpert data (CSV of Parquet)."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)

    werkexpert_cols = list(dict.fromkeys(KOLOM_MAPPING.values()))
    beschikbaar = [c for c in werkexpert_cols if c in df.columns]
    df = df[beschikbaar]
    df["bron"] = "werkxpert"

    log.info("WerkExpert geladen: %s", df.shape)
    return df


def bereken_suitability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Bereken genormaliseerde suitability score per bedrijf."""
    df["norm_ervaring_bewerking"] = df.groupby("con")["med_ervaring_bewerking"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df["norm_klant_bezoeken"] = df.groupby("con")["med_klant_bezoeken"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df["suitability_score"] = (
        (0.6 * df["norm_ervaring_bewerking"] + 0.4 * df["norm_klant_bezoeken"])
        .clip(0, 1)
        .astype(np.float32)
    )

    # Tussenkolommen opruimen
    df = df.drop(columns=["norm_ervaring_bewerking", "norm_klant_bezoeken"])
    return df


def main(rister_path: str, werkexpert_path: str, output_dir: str) -> str:
    """Voer de volledige merge pipeline uit en return het output pad."""
    os.makedirs(output_dir, exist_ok=True)

    df_rister = load_rister(rister_path)
    df_werkexpert = load_werkexpert(werkexpert_path)

    df = pd.concat([df_werkexpert, df_rister], ignore_index=True)
    log.info("Gecombineerd: %s", df.shape)

    df = bereken_suitability_score(df)

    output_path = os.path.join(output_dir, "dataframe_gecombineerd.csv")
    df.to_csv(output_path, index=False)
    log.info("Opgeslagen: %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Merge Rister + WerkExpert datasets")
    parser.add_argument(
        "--rister-path",
        default=os.environ.get("RISTER_PATH", "data/rister.csv"),
    )
    parser.add_argument(
        "--werkexpert-path",
        default=os.environ.get("WERKEXPERT_PATH", "data/werkexpert.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "data"),
    )
    args = parser.parse_args()
    main(args.rister_path, args.werkexpert_path, args.output_dir)
