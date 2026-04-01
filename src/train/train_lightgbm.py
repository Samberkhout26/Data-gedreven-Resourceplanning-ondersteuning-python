"""Train LightGBM regressor + ranker, export ONNX, log naar MLflow.

Bron: notebooks/3.modellen/lightgbm_rister (4).ipynb
"""

import argparse
import json
import logging
import os

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import train_test_split

from src.config import CATEGORICAL, FEATURES, MLFLOW_EXPERIMENT, NUMERICAL, TARGET_RANK, TARGET_TIME

log = logging.getLogger(__name__)

GROUP_COLS = ["con", "BEWERKING_ID"]


# ── Data laden & voorbereiden ─────────────────────────────────────────────────


def load_and_prepare(input_dir: str) -> pd.DataFrame:
    """Laad gecombineerde dataset en bereid features voor."""
    path = os.path.join(input_dir, "dataframe_gecombineerd.csv")
    df = pd.read_csv(path, low_memory=False)
    df.rename(columns={"aarURENVERANTW_MEDID": "URENVERANTW_MEDID"}, inplace=True)
    log.info("Data geladen: %d rijen, %d kolommen", len(df), len(df.columns))

    for col in CATEGORICAL:
        df[col] = df[col].fillna("ONBEKEND").astype("category")
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ── Optuna tuning ─────────────────────────────────────────────────────────────


def _get_cached_params(client: MlflowClient, experiment_id: str, prefix: str) -> dict | None:
    """Haal eerder gevonden Optuna params op uit MLflow."""
    filter_str = f"tags.mlflow.runName LIKE '{prefix}%'"
    order = "metrics.val_mae ASC" if "reg" in prefix else "metrics.val_ndcg_at_3 DESC"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_str,
        order_by=[order],
        max_results=1,
    )
    if not runs:
        return None

    p = runs[0].data.params
    try:
        return {
            "n_estimators": int(p["n_estimators"]),
            "learning_rate": float(p["learning_rate"]),
            "max_depth": int(p["max_depth"]),
            "num_leaves": int(p["num_leaves"]),
            "min_child_samples": int(p["min_child_samples"]),
            "subsample": float(p["subsample"]),
            "colsample_bytree": float(p["colsample_bytree"]),
            "reg_alpha": float(p["reg_alpha"]),
            "reg_lambda": float(p["reg_lambda"]),
        }
    except (KeyError, ValueError):
        return None


def tune_regressor(
    X_train, y_train, X_val, y_val, force: bool, client: MlflowClient, experiment_id: str
) -> dict:
    """Tune regressor hyperparameters met Optuna of gebruik cached params."""
    base = {"objective": "huber", "metric": "mae", "verbosity": -1, "n_jobs": -1}

    if not force and experiment_id:
        cached = _get_cached_params(client, experiment_id, "reg_trial_")
        if cached:
            log.info("Regressor params uit cache geladen")
            return {**base, **cached}

    log.info("Optuna regressor tuning gestart (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            **base,
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        with mlflow.start_run(run_name=f"reg_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                categorical_feature=CATEGORICAL,
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mlflow.log_metrics(
                {"val_mae": mae, "val_rmse": rmse, "best_iteration": model.best_iteration_}
            )
        return mae

    with mlflow.start_run(run_name="optuna_regressor"):
        study = optuna.create_study(direction="minimize", study_name="lgbm_regressor")
        study.optimize(objective, n_trials=50, show_progress_bar=True)

    log.info("Beste MAE: %.4f uur (%.1f min)", study.best_value, study.best_value * 60)
    return {**base, **study.best_params}


def tune_ranker(
    X_train,
    y_train,
    train_groups,
    X_val,
    y_val_int,
    y_val_float,
    val_groups,
    force: bool,
    client: MlflowClient,
    experiment_id: str,
) -> dict:
    """Tune ranker hyperparameters met Optuna of gebruik cached params."""
    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "verbosity": -1,
        "n_jobs": -1,
    }

    if not force and experiment_id:
        cached = _get_cached_params(client, experiment_id, "rank_trial_")
        if cached:
            log.info("Ranker params uit cache geladen")
            return {**base, **cached}

    log.info("Optuna ranker tuning gestart (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            **base,
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        with mlflow.start_run(run_name=f"rank_trial_{trial.number}", nested=True):
            mlflow.log_params({k: v for k, v in params.items() if k != "ndcg_eval_at"})
            model = lgb.LGBMRanker(**params)
            model.fit(
                X_train,
                y_train,
                group=train_groups,
                eval_set=[(X_val, y_val_int)],
                eval_group=[val_groups],
                categorical_feature=CATEGORICAL,
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_val)
            ndcg_scores_list = []
            offset = 0
            for size in val_groups:
                if size < 2:
                    offset += size
                    continue
                g_true = y_val_float[offset : offset + size]
                g_pred = preds[offset : offset + size]
                ndcg_scores_list.append(ndcg_score([g_true], [g_pred], k=3))
                offset += size
            mean_ndcg = np.mean(ndcg_scores_list)
            mlflow.log_metrics(
                {"val_ndcg_at_3": mean_ndcg, "best_iteration": model.best_iteration_}
            )
        return mean_ndcg

    with mlflow.start_run(run_name="optuna_ranker"):
        study = optuna.create_study(direction="maximize", study_name="lgbm_ranker")
        study.optimize(objective, n_trials=50, show_progress_bar=True)

    log.info("Beste NDCG@3: %.4f", study.best_value)
    return {**base, **study.best_params}


# ── Training ──────────────────────────────────────────────────────────────────


def train_regressor(params: dict, X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
    """Train het finale regressiemodel."""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    log.info("Regressor getraind, beste iteratie: %d", model.best_iteration_)
    return model


def train_ranker(
    params: dict, X_train, y_train, train_groups, X_val, y_val, val_groups
) -> lgb.LGBMRanker:
    """Train het finale ranking model."""
    model = lgb.LGBMRanker(**params)
    model.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    log.info("Ranker getraind, beste iteratie: %d", model.best_iteration_)
    return model


# ── Fine-tuning per database ─────────────────────────────────────────────────


def finetune_per_db(
    df: pd.DataFrame,
    reg_model: lgb.LGBMRegressor,
    rank_model: lgb.LGBMRanker,
) -> tuple[dict, dict]:
    """Fine-tune modellen per database (min 500 rijen)."""
    finetuned_reg, finetuned_rank = {}, {}
    db_col = "con"

    for db_id in df[db_col].unique():
        db_mask = df[db_col] == db_id
        db_indices = df.index[db_mask].values
        if len(db_indices) < 500:
            log.info("%s: %d rijen — te weinig, gebruikt basismodel", db_id, len(db_indices))
            continue

        tr, va = train_test_split(db_indices, test_size=0.2, random_state=42)
        X_ft_train, X_ft_val = df.loc[tr, FEATURES], df.loc[va, FEATURES]

        # Fine-tune regressor
        ft_reg = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            verbosity=-1,
            n_jobs=-1,
            n_estimators=100,
            learning_rate=0.01,
        )
        ft_reg.fit(
            X_ft_train,
            df.loc[tr, TARGET_TIME].values,
            eval_set=[(X_ft_val, df.loc[va, TARGET_TIME].values)],
            categorical_feature=CATEGORICAL,
            callbacks=[lgb.early_stopping(20, verbose=False)],
            init_model=reg_model,
        )
        ft_mae = mean_absolute_error(df.loc[va, TARGET_TIME], ft_reg.predict(X_ft_val))
        finetuned_reg[db_id] = ft_reg

        # Fine-tune ranker (als genoeg groepen)
        db_sorted = df.loc[db_mask].sort_values("_group_key")
        unique_groups = db_sorted["_group_key"].unique()
        if len(unique_groups) < 5:
            log.info(
                "%s: %d rijen, alleen regressor fine-tuned (MAE: %.4f)",
                db_id,
                len(db_indices),
                ft_mae,
            )
            continue

        db_tr_keys, db_va_keys = train_test_split(unique_groups, test_size=0.2, random_state=42)
        db_tr_mask = db_sorted["_group_key"].isin(set(db_tr_keys))
        db_va_mask = db_sorted["_group_key"].isin(set(db_va_keys))

        ft_rank = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[3, 5],
            verbosity=-1,
            n_jobs=-1,
            n_estimators=100,
            learning_rate=0.01,
        )
        ft_rank.fit(
            db_sorted.loc[db_tr_mask, FEATURES],
            db_sorted.loc[db_tr_mask, "_rank_label"].values,
            group=db_sorted.loc[db_tr_mask].groupby("_group_key").size().values,
            eval_set=[
                (
                    db_sorted.loc[db_va_mask, FEATURES],
                    db_sorted.loc[db_va_mask, "_rank_label"].values,
                )
            ],
            eval_group=[db_sorted.loc[db_va_mask].groupby("_group_key").size().values],
            categorical_feature=CATEGORICAL,
            callbacks=[lgb.early_stopping(20, verbose=False)],
            init_model=rank_model,
        )
        finetuned_rank[db_id] = ft_rank
        log.info("%s: %d rijen — fine-tuned, val MAE: %.4f uur", db_id, len(db_indices), ft_mae)

    return finetuned_reg, finetuned_rank


# ── Evaluatie ─────────────────────────────────────────────────────────────────


def evaluate_regressor(model, X_val, y_val) -> dict:
    """Evalueer regressiemodel."""
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    log.info("Regressor — MAE: %.4f uur (%.1f min), RMSE: %.4f, R2: %.4f", mae, mae * 60, rmse, r2)
    return {"time_mae": mae, "time_rmse": rmse, "time_r2": r2}


def evaluate_ranker(model, X_val, y_val_float, val_groups) -> dict:
    """Evalueer ranking model."""
    preds = model.predict(X_val)
    ndcg3_scores, ndcg5_scores = [], []
    offset = 0
    for size in val_groups:
        if size < 2:
            offset += size
            continue
        g_true = y_val_float[offset : offset + size]
        g_pred = preds[offset : offset + size]
        ndcg3_scores.append(ndcg_score([g_true], [g_pred], k=3))
        ndcg5_scores.append(ndcg_score([g_true], [g_pred], k=5))
        offset += size
    log.info("Ranker — NDCG@3: %.4f, NDCG@5: %.4f", np.mean(ndcg3_scores), np.mean(ndcg5_scores))
    return {"ndcg_at_3": np.mean(ndcg3_scores), "ndcg_at_5": np.mean(ndcg5_scores)}


def top_k_hit_rate(
    finetuned_rank: dict,
    val_idx,
    df: pd.DataFrame,
    base_rank_model,
    k: int = 3,
    n_samples: int = 500,
) -> float:
    """Bereken top-K hit rate op fine-tuned modellen."""
    hits, total = 0, 0
    sample_indices = np.random.choice(len(val_idx), min(n_samples, len(val_idx)), replace=False)

    for i in sample_indices:
        orig_idx = val_idx[i]
        row = df.iloc[orig_idx]
        db_id = row["con"]
        werkelijke_med = row["URENVERANTW_MEDID"]

        db_mask = df["con"] == db_id
        alle_medewerkers = df.loc[db_mask, "URENVERANTW_MEDID"].unique()
        if len(alle_medewerkers) < 2:
            continue

        ranker = finetuned_rank.get(db_id, base_rank_model)
        batch = pd.DataFrame([row[FEATURES]] * len(alle_medewerkers))
        batch.columns = FEATURES
        batch["URENVERANTW_MEDID"] = alle_medewerkers
        for col in CATEGORICAL:
            batch[col] = batch[col].astype("category")

        scores = ranker.predict(batch)
        top_k_meds = alle_medewerkers[np.argsort(scores)[-k:]]
        if werkelijke_med in top_k_meds:
            hits += 1
        total += 1

    rate = hits / total if total > 0 else 0
    log.info("Top-%d hit rate: %d/%d = %.1f%%", k, hits, total, rate * 100)
    return rate


# ── Model opslag & ONNX export ───────────────────────────────────────────────


def save_models(
    output_dir: str,
    reg_model,
    rank_model,
    finetuned_reg: dict,
    finetuned_rank: dict,
    reg_params: dict,
    rank_params: dict,
    X_val,
):
    """Sla modellen op als .txt, .onnx en metadata/cat_codes JSON."""
    models_dir = os.path.join(output_dir, "models")
    onnx_dir = os.path.join(output_dir, "models_onnx")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)

    # LightGBM .txt formaat
    reg_model.booster_.save_model(os.path.join(models_dir, "lgbm_regressor.txt"))
    rank_model.booster_.save_model(os.path.join(models_dir, "lgbm_ranker.txt"))

    for db_id, ft_reg in finetuned_reg.items():
        ft_reg.booster_.save_model(os.path.join(models_dir, f"lgbm_regressor_{db_id}.txt"))
        if db_id in finetuned_rank:
            finetuned_rank[db_id].booster_.save_model(
                os.path.join(models_dir, f"lgbm_ranker_{db_id}.txt")
            )

    # Metadata
    metadata = {
        "categorical_features": CATEGORICAL,
        "numerical_features": NUMERICAL,
        "all_features": FEATURES,
        "target_time": TARGET_TIME,
        "target_rank": TARGET_RANK,
        "best_reg_params": reg_params,
        "best_rank_params": {k: v for k, v in rank_params.items() if k != "ndcg_eval_at"},
        "finetuned_databases": list(finetuned_reg.keys()),
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Cat codes (voor C# API)
    cat_code_mapping = {}
    for col in CATEGORICAL:
        if hasattr(X_val[col], "cat"):
            cat_code_mapping[col] = dict(
                zip(X_val[col].cat.categories.tolist(), range(len(X_val[col].cat.categories)))
            )
    with open(os.path.join(models_dir, "cat_codes.json"), "w") as f:
        json.dump(cat_code_mapping, f, indent=2, default=str)

    # ONNX export
    try:
        import onnxmltools
        from skl2onnx.common.data_types import FloatTensorType

        n_features = len(FEATURES)
        initial_types = [("features", FloatTensorType([None, n_features]))]

        onnx_reg = onnxmltools.convert_lightgbm(
            reg_model, initial_types=initial_types, target_opset=15
        )
        onnxmltools.utils.save_model(onnx_reg, os.path.join(onnx_dir, "lgbm_regressor.onnx"))

        onnx_rank = onnxmltools.convert_lightgbm(
            rank_model, initial_types=initial_types, target_opset=15
        )
        onnxmltools.utils.save_model(onnx_rank, os.path.join(onnx_dir, "lgbm_ranker.onnx"))

        for db_id in finetuned_reg:
            try:
                ft_onnx_reg = onnxmltools.convert_lightgbm(
                    finetuned_reg[db_id], initial_types=initial_types, target_opset=15
                )
                onnxmltools.utils.save_model(
                    ft_onnx_reg, os.path.join(onnx_dir, f"lgbm_regressor_{db_id}.onnx")
                )
                if db_id in finetuned_rank:
                    ft_onnx_rank = onnxmltools.convert_lightgbm(
                        finetuned_rank[db_id], initial_types=initial_types, target_opset=15
                    )
                    onnxmltools.utils.save_model(
                        ft_onnx_rank, os.path.join(onnx_dir, f"lgbm_ranker_{db_id}.onnx")
                    )
                log.info("ONNX %s: OK", db_id)
            except Exception as e:
                log.warning("ONNX %s: FOUT — %s", db_id, e)

        log.info("ONNX export voltooid")
    except ImportError:
        log.warning("onnxmltools niet beschikbaar, ONNX export overgeslagen")

    return models_dir, onnx_dir


# ── MLflow logging ────────────────────────────────────────────────────────────


def log_to_mlflow(
    reg_params,
    rank_params,
    reg_metrics,
    rank_metrics,
    hit_rate,
    models_dir,
    onnx_dir,
    df_len,
):
    """Log alles naar MLflow en registreer modellen als MAE verbetert."""
    client = MlflowClient()

    with mlflow.start_run(run_name="lightgbm_final") as run:
        mlflow.log_params({f"reg_{k}": v for k, v in reg_params.items()})
        mlflow.log_params({f"rank_{k}": v for k, v in rank_params.items() if k != "ndcg_eval_at"})
        mlflow.log_metrics(
            {**reg_metrics, **rank_metrics, "top3_hit_rate": hit_rate, "dataset_rows": df_len}
        )

        # Log artifacts
        for fname in os.listdir(models_dir):
            mlflow.log_artifact(os.path.join(models_dir, fname))
        if os.path.isdir(onnx_dir):
            for fname in os.listdir(onnx_dir):
                mlflow.log_artifact(os.path.join(onnx_dir, fname), artifact_path="onnx")

        run_id = run.info.run_id

    # Conditional registration
    mae = reg_metrics["time_mae"]

    def get_best_registry_mae(model_name):
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            maes = []
            for v in versions:
                metrics = client.get_run(v.run_id).data.metrics
                if "time_mae" in metrics:
                    maes.append(metrics["time_mae"])
            return min(maes) if maes else float("inf")
        except Exception:
            return float("inf")

    best_in_registry = get_best_registry_mae("rister-lgbm-regressor")
    if mae < best_in_registry:
        mlflow.register_model(f"runs:/{run_id}/lgbm_regressor", "rister-lgbm-regressor")
        mlflow.register_model(f"runs:/{run_id}/lgbm_ranker", "rister-lgbm-ranker")
        log.info("Nieuwe beste versie geregistreerd! MAE: %.4f (was: %.4f)", mae, best_in_registry)
    else:
        log.info("Niet geregistreerd — MAE %.4f niet beter dan %.4f", mae, best_in_registry)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(input_dir: str, output_dir: str, force_tuning: bool = False):
    """Voer de volledige training pipeline uit."""
    os.makedirs(output_dir, exist_ok=True)

    # MLflow setup (Azure ML configureert dit automatisch)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", os.environ.get("MLFLOW_URI", ""))
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else None

    # Data laden
    df = load_and_prepare(input_dir)

    # ── Regressor split ───────────────────────────────────────────────────
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    X_train, X_val = df.loc[train_idx, FEATURES], df.loc[val_idx, FEATURES]
    y_time_train = df.loc[train_idx, TARGET_TIME].values
    y_time_val = df.loc[val_idx, TARGET_TIME].values

    # Tune & train regressor
    reg_params = tune_regressor(
        X_train, y_time_train, X_val, y_time_val, force_tuning, client, experiment_id
    )
    reg_model = train_regressor(reg_params, X_train, y_time_train, X_val, y_time_val)
    reg_metrics = evaluate_regressor(reg_model, X_val, y_time_val)

    # ── Ranker split (group-based) ────────────────────────────────────────
    df["_group_key"] = df[GROUP_COLS].astype(str).agg("__".join, axis=1)
    bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
    df["_rank_label"] = pd.cut(df[TARGET_RANK], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)

    df_sorted = df.sort_values("_group_key").reset_index(drop=True)
    group_keys = list(df_sorted.groupby("_group_key").groups.keys())
    train_keys, val_keys = train_test_split(group_keys, test_size=0.2, random_state=42)

    train_mask = df_sorted["_group_key"].isin(set(train_keys))
    val_mask = df_sorted["_group_key"].isin(set(val_keys))

    X_rank_train = df_sorted.loc[train_mask, FEATURES]
    X_rank_val = df_sorted.loc[val_mask, FEATURES]
    y_rank_train = df_sorted.loc[train_mask, "_rank_label"].values
    y_rank_val_int = df_sorted.loc[val_mask, "_rank_label"].values
    y_rank_val_float = df_sorted.loc[val_mask, TARGET_RANK].values
    train_groups = df_sorted.loc[train_mask].groupby("_group_key").size().values
    val_groups = df_sorted.loc[val_mask].groupby("_group_key").size().values

    # Tune & train ranker
    rank_params = tune_ranker(
        X_rank_train,
        y_rank_train,
        train_groups,
        X_rank_val,
        y_rank_val_int,
        y_rank_val_float,
        val_groups,
        force_tuning,
        client,
        experiment_id,
    )
    rank_model = train_ranker(
        rank_params,
        X_rank_train,
        y_rank_train,
        train_groups,
        X_rank_val,
        y_rank_val_int,
        val_groups,
    )
    rank_metrics = evaluate_ranker(rank_model, X_rank_val, y_rank_val_float, val_groups)

    # ── Fine-tuning ───────────────────────────────────────────────────────
    finetuned_reg, finetuned_rank = finetune_per_db(df, reg_model, rank_model)

    # ── Top-K hit rate ────────────────────────────────────────────────────
    hit_rate = top_k_hit_rate(finetuned_rank, val_idx, df, rank_model)

    # ── Opslaan ───────────────────────────────────────────────────────────
    models_dir, onnx_dir = save_models(
        output_dir,
        reg_model,
        rank_model,
        finetuned_reg,
        finetuned_rank,
        reg_params,
        rank_params,
        X_val,
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    log_to_mlflow(
        reg_params,
        rank_params,
        reg_metrics,
        rank_metrics,
        hit_rate,
        models_dir,
        onnx_dir,
        len(df),
    )

    log.info("Training pipeline voltooid")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Train LightGBM regressor + ranker")
    parser.add_argument("--input-dir", default=os.environ.get("INPUT_DIR", "data"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"))
    parser.add_argument(
        "--force-tuning", action="store_true", help="Forceer Optuna tuning (model drift)"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.force_tuning)
