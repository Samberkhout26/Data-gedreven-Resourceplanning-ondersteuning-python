"""
Training: LightGBM regressor (tijdvoorspelling) + ranker (medewerkergeschiktheid)
- Hyperparameter tuning via Optuna (hervat vanuit MLflow als er al runs zijn)
- Fine-tuning per database
- ONNX-export
- MLflow logging + model registry (alleen registreren als beter dan huidige best)
"""

import json
import logging
import os
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.onnx
import numpy as np
import onnxmltools
import onnxruntime as ort
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from onnxmltools.convert.lightgbm.convert import convert as lgbm_to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MLFLOW_URI, MODELS_DIR, MODELS_ONNX_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Feature configuratie ---
CATEGORICAL = [
    "URENVERANTW_MEDID", "BEWERKING_ID", "DIENST_ART_ID", "RELATIE_ID",
    "REL_POSTCODE", "DIENST_ART_OMS", "MACH_OMS", "con", "bron",
    "EquipmentGroupTypes", "planninggroupsname",
]
NUMERICAL = [
    "lat", "lon",
    "dag_sin", "dag_cos", "maand_sin", "maand_cos", "week_sin", "week_cos",
    "med_std_tijd", "med_aantal_opdrachten", "med_ervaring_bewerking", "med_gem_tijd",
    "taak_gem", "med_klant_bezoeken", "med_klant_ratio", "med_klant_snelheid",
    "med_bewerking_snelheid", "med_klant_gem_tijd", "med_bewerking_gem_tijd", "med_totaal_opdrachten",
    "hoeveelheid_volume", "hoeveelheid_gewicht", "hoeveelheid_stuks", "hoeveelheid_aanwezig",
    "hoeveelheid_baal",
]
FEATURES = CATEGORICAL + NUMERICAL
TARGET_TIME = "REAL_WORKED_TIME"
TARGET_RANK = "suitability_score"
MLFLOW_EXPERIMENT = "rister-lightgbm-v1-plushoeveelheid"


def _load_best_params_from_mlflow(client, experiment_id, prefix, sort_metric, ascending=True):
    """Laad beste hyperparameters uit bestaande MLflow-runs."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{prefix}%'",
        order_by=[f"metrics.{sort_metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    if not runs:
        return None
    best = runs[0]
    return {k.replace(f"{prefix}_", ""): v for k, v in best.data.params.items()}


def _prepare_data(df):
    """Type conversie en train/val split."""
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("ONBEKEND").astype("category")
        else:
            df[col] = pd.Categorical(["ONBEKEND"] * len(df))

    for col in NUMERICAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features]
    y_time = df[TARGET_TIME]
    y_rank = df[TARGET_RANK]

    X_train, X_val, y_time_train, y_time_val, y_rank_train, y_rank_val = train_test_split(
        X, y_time, y_rank, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_time_train, y_time_val, y_rank_train, y_rank_val, available_features


def _train_regressor(X_train, X_val, y_train, y_val, best_params=None):
    if best_params is None:
        best_params = {
            "objective": "huber", "metric": "mae", "verbosity": -1, "n_jobs": -1,
            "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
            "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
        }
    else:
        best_params.update({"objective": "huber", "metric": "mae", "verbosity": -1, "n_jobs": -1})

    cat_cols = [c for c in CATEGORICAL if c in X_train.columns]
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def _train_ranker(X_train, X_val, y_rank_train, y_rank_val, best_params=None):
    GROUP_COLS = ["con", "BEWERKING_ID"]
    available_group = [c for c in GROUP_COLS if c in X_train.columns]

    def make_groups_and_labels(X, y_rank):
        df_tmp = X.copy()
        df_tmp["_rank_label"] = pd.cut(
            y_rank.values, bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001], labels=[0, 1, 2, 3, 4]
        ).astype(int)
        if available_group:
            df_tmp["_group_key"] = df_tmp[available_group].astype(str).agg("_".join, axis=1)
        else:
            df_tmp["_group_key"] = "default"
        df_tmp = df_tmp.sort_values("_group_key").reset_index(drop=True)
        groups = df_tmp.groupby("_group_key", sort=False).size().values
        labels = df_tmp["_rank_label"].values
        X_sorted = df_tmp.drop(columns=["_rank_label", "_group_key"])
        return X_sorted, labels, groups

    X_rank_train, y_rank_train_sorted, train_groups = make_groups_and_labels(X_train, y_rank_train)
    X_rank_val, y_rank_val_sorted, val_groups = make_groups_and_labels(X_val, y_rank_val)

    if best_params is None:
        best_params = {
            "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3, 5],
            "verbosity": -1, "n_jobs": -1, "n_estimators": 300, "learning_rate": 0.05,
            "max_depth": 6, "num_leaves": 63, "min_child_samples": 20,
        }
    else:
        best_params.update({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3, 5],
                            "verbosity": -1, "n_jobs": -1})

    cat_cols = [c for c in CATEGORICAL if c in X_rank_train.columns]
    model = lgb.LGBMRanker(**best_params)
    model.fit(
        X_rank_train, y_rank_train_sorted,
        group=train_groups,
        eval_set=[(X_rank_val, y_rank_val_sorted)],
        eval_group=[val_groups],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model, X_rank_val, y_rank_val_sorted, val_groups


def _evaluate_regressor(model, X_val, y_val):
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    r2 = r2_score(y_val, preds)
    log.info(f"Regressor — MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}")
    return mae, rmse, r2


def _evaluate_ranker(model, X_rank_val, y_rank_val_sorted, val_groups):
    preds = model.predict(X_rank_val)
    idx = 0
    ndcg3_scores, ndcg5_scores = [], []
    for g in val_groups:
        true_g = y_rank_val_sorted[idx:idx + g]
        pred_g = preds[idx:idx + g]
        if len(true_g) >= 2:
            ndcg3_scores.append(ndcg_score([true_g], [pred_g], k=3))
            ndcg5_scores.append(ndcg_score([true_g], [pred_g], k=5))
        idx += g
    ndcg3 = float(np.mean(ndcg3_scores)) if ndcg3_scores else 0.0
    ndcg5 = float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0
    log.info(f"Ranker — NDCG@3: {ndcg3:.4f}  NDCG@5: {ndcg5:.4f}")
    return ndcg3, ndcg5


def _to_onnx(model, n_features, name):
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    return lgbm_to_onnx(model.booster_, initial_types=initial_type, target_opset=15, name=name)


def _finetune_per_db(df, reg_model, rank_model, available_features):
    finetuned_reg, finetuned_rank = {}, {}
    cat_cols = [c for c in CATEGORICAL if c in available_features]

    for db_id in df["con"].dropna().unique():
        df_db = df[df["con"] == db_id]
        if len(df_db) < 500:
            continue

        X_db = df_db[[f for f in available_features if f in df_db.columns]]
        y_db_time = df_db[TARGET_TIME]

        # Regressor fine-tune
        ft_reg = lgb.LGBMRegressor(objective="huber", metric="mae", n_estimators=100, learning_rate=0.01)
        ft_reg.fit(
            X_db, y_db_time,
            categorical_feature=cat_cols,
            init_model=reg_model.booster_,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        finetuned_reg[db_id] = ft_reg
        log.info(f"Fine-tuned regressor: {db_id}")

        # Ranker fine-tune (alleen als genoeg groepen)
        if len(df_db["BEWERKING_ID"].dropna().unique()) >= 5 if "BEWERKING_ID" in df_db.columns else False:
            ft_rank = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=100, learning_rate=0.01)
            df_db_sorted = df_db.sort_values(["con", "BEWERKING_ID"] if "BEWERKING_ID" in df_db.columns else ["con"])
            groups_db = df_db_sorted.groupby(
                ["con", "BEWERKING_ID"] if "BEWERKING_ID" in df_db.columns else ["con"]
            ).size().values
            labels_db = pd.cut(
                df_db_sorted[TARGET_RANK].values,
                bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001], labels=[0, 1, 2, 3, 4]
            ).astype(int)
            ft_rank.fit(
                df_db_sorted[[f for f in available_features if f in df_db_sorted.columns]],
                labels_db,
                group=groups_db,
                categorical_feature=cat_cols,
                init_model=rank_model.booster_,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )
            finetuned_rank[db_id] = ft_rank
            log.info(f"Fine-tuned ranker: {db_id}")

    return finetuned_reg, finetuned_rank


def voorspel_per_medewerker(reg_model, rank_model, taak_rij, medewerkers, features):
    """Voorspel werktijd en geschiktheid voor een lijst medewerkers op één taak."""
    batch = pd.concat([taak_rij.assign(**{"URENVERANTW_MEDID": m}) for m in medewerkers], ignore_index=True)
    available = [f for f in features if f in batch.columns]
    tijden = np.clip(reg_model.predict(batch[available]), 0, None).round(2)
    scores = rank_model.predict(batch[available]).round(3)
    return pd.DataFrame({
        "medewerker": medewerkers,
        "voorspelde_tijd": tijden,
        "geschiktheid": scores,
    }).sort_values("geschiktheid", ascending=False).reset_index(drop=True)


def run():
    log.info("Start LightGBM training")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else None

    # --- Data laden ---
    data_path = PROCESSED_DIR / "dataframe_gecombineerd.csv"
    log.info(f"Laden {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    log.info(f"Dataset: {len(df)} rijen")

    X_train, X_val, y_time_train, y_time_val, y_rank_train, y_rank_val, available_features = _prepare_data(df)

    # --- Beste params ophalen uit MLflow (als al runs bestaan) ---
    reg_params = None
    rank_params = None
    if experiment_id:
        reg_params = _load_best_params_from_mlflow(client, experiment_id, "reg_trial", "val_mae", ascending=True)
        rank_params = _load_best_params_from_mlflow(client, experiment_id, "rank_trial", "val_ndcg_at_3", ascending=False)

    if reg_params:
        log.info("Regressor params geladen uit MLflow")
    if rank_params:
        log.info("Ranker params geladen uit MLflow")

    # --- Modellen trainen ---
    log.info("Trainen regressor")
    reg_model = _train_regressor(X_train, X_val, y_time_train, y_time_val, reg_params)

    log.info("Trainen ranker")
    rank_model, X_rank_val, y_rank_val_sorted, val_groups = _train_ranker(
        X_train, X_val, y_rank_train, y_rank_val, rank_params
    )

    # --- Evaluatie ---
    mae, rmse, r2 = _evaluate_regressor(reg_model, X_val, y_time_val)
    ndcg3, ndcg5 = _evaluate_ranker(rank_model, X_rank_val, y_rank_val_sorted, val_groups)

    # --- Fine-tuning per database ---
    log.info("Fine-tuning per database")
    finetuned_reg, finetuned_rank = _finetune_per_db(df, reg_model, rank_model, available_features)

    # --- Modellen opslaan ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ONNX_DIR.mkdir(parents=True, exist_ok=True)

    reg_model.booster_.save_model(str(MODELS_DIR / "lgbm_regressor.txt"))
    rank_model.booster_.save_model(str(MODELS_DIR / "lgbm_ranker.txt"))

    n_features = len(available_features)
    onnx_reg = _to_onnx(reg_model, n_features, "lgbm_regressor")
    onnx_rank = _to_onnx(rank_model, n_features, "lgbm_ranker")

    with open(MODELS_ONNX_DIR / "lgbm_regressor.onnx", "wb") as f:
        f.write(onnx_reg.SerializeToString())
    with open(MODELS_ONNX_DIR / "lgbm_ranker.onnx", "wb") as f:
        f.write(onnx_rank.SerializeToString())

    # Ook opslaan onder de namen die de C# API verwacht
    with open(MODELS_ONNX_DIR / "regressor.onnx", "wb") as f:
        f.write(onnx_reg.SerializeToString())
    with open(MODELS_ONNX_DIR / "ranker.onnx", "wb") as f:
        f.write(onnx_rank.SerializeToString())
    log.info("Basis ONNX-modellen opgeslagen")

    # Fine-tuned modellen opslaan
    for db_id, ft in finetuned_reg.items():
        ft.booster_.save_model(str(MODELS_DIR / f"lgbm_regressor_{db_id}.txt"))
        try:
            onnx_ft = _to_onnx(ft, n_features, f"lgbm_regressor_{db_id}")
            with open(MODELS_ONNX_DIR / f"lgbm_regressor_{db_id}.onnx", "wb") as f:
                f.write(onnx_ft.SerializeToString())
        except Exception as e:
            log.warning(f"ONNX-conversie regressor {db_id} mislukt: {e}")

    for db_id, ft in finetuned_rank.items():
        ft.booster_.save_model(str(MODELS_DIR / f"lgbm_ranker_{db_id}.txt"))
        try:
            onnx_ft = _to_onnx(ft, n_features, f"lgbm_ranker_{db_id}")
            with open(MODELS_ONNX_DIR / f"lgbm_ranker_{db_id}.onnx", "wb") as f:
                f.write(onnx_ft.SerializeToString())
        except Exception as e:
            log.warning(f"ONNX-conversie ranker {db_id} mislukt: {e}")

    # Metadata + cat_codes opslaan
    cat_codes = {}
    for col in CATEGORICAL:
        if col in X_train.columns and hasattr(X_train[col], "cat"):
            cat_codes[col] = {str(v): i for i, v in enumerate(X_train[col].cat.categories)}

    metadata = {
        "categorical_features": CATEGORICAL,
        "numerical_features": NUMERICAL,
        "all_features": available_features,
        "target_time": TARGET_TIME,
        "target_rank": TARGET_RANK,
        "finetuned_databases": list(finetuned_reg.keys()),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(MODELS_DIR / "cat_codes.json", "w") as f:
        json.dump(cat_codes, f, indent=2)

    log.info("Metadata en cat_codes opgeslagen")

    # --- MLflow logging ---
    with mlflow.start_run(run_name="lightgbm_final") as run:
        mlflow.log_params({f"reg_{k}": v for k, v in (reg_params or {}).items()})
        mlflow.log_params({k: v for k, v in (rank_params or {}).items()
                          if k not in ["ndcg_eval_at"]})
        mlflow.log_metrics({
            "time_mae": mae,
            "time_rmse": rmse,
            "time_r2": r2,
            "ndcg_at_3": ndcg3,
            "ndcg_at_5": ndcg5,
            "dataset_rows": len(df),
        })
        mlflow.onnx.log_model(onnx_reg, name="lgbm_regressor")
        mlflow.onnx.log_model(onnx_rank, name="lgbm_ranker")
        mlflow.log_artifact(str(MODELS_DIR / "lgbm_regressor.txt"))
        mlflow.log_artifact(str(MODELS_DIR / "lgbm_ranker.txt"))
        mlflow.log_artifact(str(MODELS_DIR / "metadata.json"))
        mlflow.log_artifact(str(MODELS_DIR / "cat_codes.json"))

        run_id = run.info.run_id

    log.info(f"MLflow run: {run_id}")

    # --- Model registry: alleen registreren als beter ---
    try:
        versions = client.search_model_versions("name='rister-lgbm-regressor'")
        best_mae = min(
            float(client.get_run(v.run_id).data.metrics.get("time_mae", 9999))
            for v in versions
        ) if versions else 9999
    except Exception:
        best_mae = 9999

    if mae < best_mae:
        client.create_registered_model("rister-lgbm-regressor") if not versions else None
        client.create_model_version(
            name="rister-lgbm-regressor",
            source=f"runs:/{run_id}/lgbm_regressor",
            run_id=run_id,
        )
        client.create_model_version(
            name="rister-lgbm-ranker",
            source=f"runs:/{run_id}/lgbm_ranker",
            run_id=run_id,
        )
        log.info(f"Nieuw model geregistreerd (MAE {mae:.3f} < {best_mae:.3f})")
    else:
        log.info(f"Model niet geregistreerd (MAE {mae:.3f} >= huidige best {best_mae:.3f})")

    log.info("Training klaar")


if __name__ == "__main__":
    run()
