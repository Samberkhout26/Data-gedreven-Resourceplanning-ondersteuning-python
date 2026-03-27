"""
Prefect flow: Rister ML Pipeline

Elke stap is een aparte task met eigen logging, retry-logica en status in de Prefect UI.

Starten:
    python src/flows/pipeline_flow.py          # eenmalig draaien
    prefect server start                        # open UI op http://127.0.0.1:4200

Deployen met schema (wekelijks):
    python src/flows/deploy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import timedelta

from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash


@task(
    name="Rister dataprep",
    description="Laadt data uit PostgreSQL en slaat rister.csv op",
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
)
def stap_rister():
    logger = get_run_logger()
    logger.info("Start Rister dataprep")
    from dataprep.rister import run
    run()
    logger.info("Rister dataprep klaar")


@task(
    name="WerkExpert dataprep",
    description="Laadt data uit 26 Firebird-databases en slaat werkexpert.csv op",
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
)
def stap_werkexpert():
    logger = get_run_logger()
    logger.info("Start WerkExpert dataprep")
    from dataprep.werkexpert import run
    run()
    logger.info("WerkExpert dataprep klaar")


@task(
    name="Merge",
    description="Combineert rister.csv + werkexpert.csv tot dataframe_gecombineerd.csv",
    retries=1,
    retry_delay_seconds=30,
)
def stap_merge():
    logger = get_run_logger()
    logger.info("Start merge")
    from dataprep.merge import run
    run()
    logger.info("Merge klaar")


@task(
    name="LightGBM training",
    description="Traint regressor + ranker, exporteert ONNX, logt naar MLflow",
    retries=1,
    retry_delay_seconds=60,
    timeout_seconds=3600,  # max 1 uur
)
def stap_training():
    logger = get_run_logger()
    logger.info("Start training")
    from train.train_lightgbm import run
    run()
    logger.info("Training klaar")


@flow(
    name="Rister ML Pipeline",
    description="Volledige pipeline: dataprep → merge → training → MLflow",
    log_prints=True,
)
def rister_pipeline(
    skip_rister: bool = False,
    skip_werkexpert: bool = False,
    skip_merge: bool = False,
    skip_train: bool = False,
):
    """
    Parameters
    ----------
    skip_rister      : Sla Rister dataprep over (bijv. als data niet veranderd is)
    skip_werkexpert  : Sla WerkExpert dataprep over
    skip_merge       : Sla merge stap over
    skip_train       : Sla training over
    """
    if not skip_rister:
        stap_rister()

    if not skip_werkexpert:
        stap_werkexpert()

    if not skip_merge:
        stap_merge()

    if not skip_train:
        stap_training()


if __name__ == "__main__":
    rister_pipeline()
