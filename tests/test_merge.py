"""
Tests voor src/dataprep/merge.py

Getest zonder database of bestandssysteem: puur de dataframe-logica.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# src/ op het pad zodat config en merge importeerbaar zijn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataprep.merge import KOLOM_MAPPING, RISTER_ONLY

# ── Helpers ───────────────────────────────────────────────────────────────────


def _maak_rister_df(n=10):
    """Minimale Rister DataFrame met alle kolommen uit KOLOM_MAPPING."""
    data = {col: [f"val_{i}" for i in range(n)] for col in KOLOM_MAPPING}
    data["EmployeeId"] = [f"EMP{i:03d}" for i in range(n)]
    data["hoeveelheid_uur"] = np.random.uniform(0.5, 8.0, n)
    data["TenantId_x"] = ["CON_A"] * (n // 2) + ["CON_B"] * (n - n // 2)
    for rister_col in RISTER_ONLY:
        data[rister_col] = [f"r_{i}" for i in range(n)]
    # Numerieke kolommen
    for col in [
        "med_ervaring_bewerking",
        "med_klant_bezoeken",
        "med_gem_tijd",
        "med_std_tijd",
        "med_aantal_opdrachten",
        "taak_gem",
        "med_klant_ratio",
        "med_klant_snelheid",
        "med_bewerking_snelheid",
        "med_klant_gem_tijd",
        "med_bewerking_gem_tijd",
        "med_totaal_opdrachten",
        "lat",
        "lon",
        "hoeveelheid_big_bag",
        "hoeveelheid_dag",
        "hoeveelheid_hectare",
        "hoeveelheid_kubieke_meter",
        "dag_sin",
        "dag_cos",
        "maand_sin",
        "maand_cos",
        "week_sin",
        "week_cos",
    ]:
        if col not in data:
            data[col] = np.random.uniform(0, 10, n)
    return pd.DataFrame(data)


def _maak_werkexpert_df(n=8):
    """Minimale WerkExpert DataFrame met doelkolomnamen."""
    doel = list(set(KOLOM_MAPPING.values()))
    data = {col: np.random.uniform(0, 5, n) for col in doel}
    data["URENVERANTW_MEDID"] = [f"MED{i:03d}" for i in range(n)]
    data["con"] = ["CON_A"] * (n // 2) + ["CON_B"] * (n - n // 2)
    data["med_ervaring_bewerking"] = np.random.uniform(0, 100, n)
    data["med_klant_bezoeken"] = np.random.uniform(0, 50, n)
    return pd.DataFrame(data)


# ── Tests: KOLOM_MAPPING ──────────────────────────────────────────────────────


def test_kolom_mapping_geen_duplicaten_in_doelnamen():
    """Elke doelkolomnaam mag maar één keer voorkomen."""
    doelen = list(KOLOM_MAPPING.values())
    assert len(doelen) == len(set(doelen)), "Duplicate doelkolomnamen in KOLOM_MAPPING"


def test_kolom_mapping_bronkolommen_uniek():
    """Elke bronkolomnaam mag maar één keer voorkomen."""
    bronnen = list(KOLOM_MAPPING.keys())
    assert len(bronnen) == len(set(bronnen)), "Duplicate bronkolomnamen in KOLOM_MAPPING"


def test_rister_only_niet_in_mapping():
    """RISTER_ONLY kolommen mogen niet ook in KOLOM_MAPPING staan."""
    for col in RISTER_ONLY:
        assert col not in KOLOM_MAPPING, f"{col} staat zowel in RISTER_ONLY als KOLOM_MAPPING"


# ── Tests: hernoemen van Rister-kolommen ─────────────────────────────────────


def test_rister_kolommen_hernoemd():
    """Na rename moeten de doelkolomnamen aanwezig zijn."""
    df = _maak_rister_df(5)
    cols_beschikbaar = {k: v for k, v in KOLOM_MAPPING.items() if k in df.columns}
    df_sel = df[list(cols_beschikbaar.keys())].rename(columns=cols_beschikbaar)
    for doel in cols_beschikbaar.values():
        assert doel in df_sel.columns, f"Doelkolom '{doel}' ontbreekt na rename"


def test_bronkolom_verdwenen_na_rename():
    """Na rename mogen de originele Rister-kolomnamen niet meer bestaan."""
    df = _maak_rister_df(5)
    cols_beschikbaar = {k: v for k, v in KOLOM_MAPPING.items() if k in df.columns}
    df_sel = df[list(cols_beschikbaar.keys())].rename(columns=cols_beschikbaar)
    for bron in cols_beschikbaar:
        # Alleen controleren als bron ≠ doel (sommige mappen naar zichzelf)
        if bron != cols_beschikbaar[bron]:
            assert bron not in df_sel.columns, f"Bronkolom '{bron}' bestaat nog na rename"


# ── Tests: bron-kolom ─────────────────────────────────────────────────────────


def test_bron_kolom_rister():
    df = _maak_rister_df(4)
    cols_beschikbaar = {k: v for k, v in KOLOM_MAPPING.items() if k in df.columns}
    df_sel = df[list(cols_beschikbaar.keys())].rename(columns=cols_beschikbaar)
    df_sel["bron"] = "rister"
    assert (df_sel["bron"] == "rister").all()


def test_bron_kolom_werkexpert():
    df = _maak_werkexpert_df(4)
    doel = list(set(KOLOM_MAPPING.values()))
    df_sel = df[[c for c in doel if c in df.columns]].copy()
    df_sel["bron"] = "werkexpert"
    assert (df_sel["bron"] == "werkexpert").all()


# ── Tests: suitability_score ──────────────────────────────────────────────────


def _bereken_suitability(df):
    """Replicatie van de suitability-logica uit merge.py."""
    df = df.copy()
    df["norm_ervaring"] = df.groupby("con")["med_ervaring_bewerking"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df["norm_klant"] = df.groupby("con")["med_klant_bezoeken"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df["suitability_score"] = (
        (0.6 * df["norm_ervaring"] + 0.4 * df["norm_klant"]).clip(0, 1).astype("float32")
    )
    return df.drop(columns=["norm_ervaring", "norm_klant"])


def test_suitability_score_range():
    """suitability_score moet altijd tussen 0 en 1 liggen."""
    df = pd.DataFrame(
        {
            "con": ["A"] * 5 + ["B"] * 5,
            "med_ervaring_bewerking": np.random.uniform(0, 100, 10),
            "med_klant_bezoeken": np.random.uniform(0, 50, 10),
        }
    )
    result = _bereken_suitability(df)
    assert result["suitability_score"].between(0, 1).all()


def test_suitability_score_max_per_groep_is_een():
    """Per 'con'-groep moet de hoogste suitability_score gelijk aan 1 zijn."""
    df = pd.DataFrame(
        {
            "con": ["A", "A", "A", "B", "B"],
            "med_ervaring_bewerking": [10.0, 20.0, 30.0, 5.0, 15.0],
            "med_klant_bezoeken": [10.0, 20.0, 30.0, 5.0, 15.0],
        }
    )
    result = _bereken_suitability(df)
    for groep in ["A", "B"]:
        max_score = result[result["con"] == groep]["suitability_score"].max()
        assert abs(max_score - 1.0) < 1e-5, (
            f"Max suitability in groep {groep} is {max_score}, verwacht 1.0"
        )


def test_suitability_score_nul_als_alle_ervaring_nul():
    """Als alle med_ervaring_bewerking = 0, moet norm_ervaring = 0."""
    df = pd.DataFrame(
        {
            "con": ["A", "A", "A"],
            "med_ervaring_bewerking": [0.0, 0.0, 0.0],
            "med_klant_bezoeken": [10.0, 20.0, 30.0],
        }
    )
    result = _bereken_suitability(df)
    # Alleen klant-component, gewicht 0.4, max norm_klant = 1 → max score = 0.4
    assert result["suitability_score"].max() <= 0.4 + 1e-5


def test_suitability_score_dtype():
    """suitability_score moet float32 zijn (zelfde als Python-training)."""
    df = pd.DataFrame(
        {
            "con": ["A"] * 3,
            "med_ervaring_bewerking": [1.0, 2.0, 3.0],
            "med_klant_bezoeken": [1.0, 2.0, 3.0],
        }
    )
    result = _bereken_suitability(df)
    assert result["suitability_score"].dtype == np.float32


# ── Tests: concat ─────────────────────────────────────────────────────────────


def test_concat_behoudt_alle_rijen():
    """Na concat moet het totale aantal rijen gelijk zijn aan de som van beide inputs."""
    df_r = _maak_rister_df(10)
    df_w = _maak_werkexpert_df(8)

    cols_beschikbaar = {k: v for k, v in KOLOM_MAPPING.items() if k in df_r.columns}
    df_r_sel = df_r[list(cols_beschikbaar.keys())].rename(columns=cols_beschikbaar)
    df_r_sel["bron"] = "rister"

    doel = list(set(KOLOM_MAPPING.values()))
    df_w_sel = df_w[[c for c in doel if c in df_w.columns]].copy()
    df_w_sel["bron"] = "werkexpert"

    df_combined = pd.concat([df_r_sel, df_w_sel], ignore_index=True)
    assert len(df_combined) == len(df_r_sel) + len(df_w_sel)


def test_concat_index_is_reset():
    """Na concat moet de index aaneengesloten zijn (0, 1, 2, ...)."""
    df_r = _maak_rister_df(3)
    df_w = _maak_werkexpert_df(3)

    cols_beschikbaar = {k: v for k, v in KOLOM_MAPPING.items() if k in df_r.columns}
    df_r_sel = df_r[list(cols_beschikbaar.keys())].rename(columns=cols_beschikbaar)
    df_r_sel["bron"] = "rister"

    doel = list(set(KOLOM_MAPPING.values()))
    df_w_sel = df_w[[c for c in doel if c in df_w.columns]].copy()
    df_w_sel["bron"] = "werkexpert"

    df_combined = pd.concat([df_r_sel, df_w_sel], ignore_index=True)
    assert list(df_combined.index) == list(range(len(df_combined)))
