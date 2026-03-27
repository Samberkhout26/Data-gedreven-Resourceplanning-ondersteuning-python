"""
Azure ML Pipeline: Rister ML
Definieert de pipeline als aaneengesloten stappen (components).

WerkExpert-data wordt NIET in Azure verwerkt (Firebird draait lokaal).
In plaats daarvan leest de pipeline werkexpert.csv uit Azure Blob Storage.
Die CSV wordt handmatig geupload via: python src/dataprep/werkexpert.py

Gebruik:
    python src/aml/pipeline.py          # eenmalig draaien (submit)
    python src/aml/deploy.py            # met kwartaalschema deployen
"""

import os
import sys
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AML_RESOURCE_GROUP, AML_SUBSCRIPTION_ID, AML_WORKSPACE, POSTGRES_URI

# Blob Storage pad naar de werkexpert CSV (geupload vanaf lokale Mac)
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "rister-data")
WERKEXPERT_BLOB_URI = os.getenv(
    "WERKEXPERT_BLOB_URI",
    f"azureml://datastores/workspaceblobstore/paths/{BLOB_CONTAINER}/processed/werkexpert.parquet",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Azure ML client ───────────────────────────────────────────────────────────

def get_ml_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AML_SUBSCRIPTION_ID,
        resource_group_name=AML_RESOURCE_GROUP,
        workspace_name=AML_WORKSPACE,
    )


# ─── Environment (Docker image) ────────────────────────────────────────────────

def get_or_create_environment(ml_client: MLClient):
    """
    Registreert de Docker image uit het project als Azure ML Environment.
    Daarna wordt dit hergebruikt en alleen opnieuw gebouwd bij Dockerfile-wijzigingen.
    """
    env = Environment(
        name="rister-pipeline-env",
        description="Rister ML pipeline omgeving",
        build=BuildContext(path=str(PROJECT_ROOT)),
        version="latest",
    )
    return ml_client.environments.create_or_update(env)


# ─── Pipeline componenten ──────────────────────────────────────────────────────

def rister_component(env_name: str):
    return command(
        name="rister_dataprep",
        display_name="Rister dataprep",
        description="PostgreSQL → rister.csv",
        command="python src/dataprep/rister.py",
        environment=env_name,
        outputs={"processed": Output(type="uri_folder")},
        environment_variables={
            "POSTGRES_URI": POSTGRES_URI,
        },
    )


def merge_component(env_name: str):
    return command(
        name="merge",
        display_name="Merge rister + werkexpert",
        description="Combineert rister (vers) + werkexpert (uit Blob) tot dataframe_gecombineerd.csv",
        command="python src/dataprep/merge.py",
        environment=env_name,
        inputs={
            "rister_data": Input(type="uri_folder"),
            "werkexpert_data": Input(type="uri_file"),   # CSV uit Blob Storage
        },
        outputs={"merged": Output(type="uri_folder")},
    )


def training_component(env_name: str):
    return command(
        name="lightgbm_training",
        display_name="LightGBM training",
        description="Traint regressor + ranker, exporteert ONNX, logt naar MLflow",
        command="python src/train/train_lightgbm.py",
        environment=env_name,
        inputs={"merged_data": Input(type="uri_folder")},
        outputs={"models": Output(type="uri_folder")},
        # Azure ML zet MLFLOW_TRACKING_URI automatisch
    )


# ─── Pipeline definitie ────────────────────────────────────────────────────────

def build_pipeline(env_name: str):
    rister = rister_component(env_name)
    merge = merge_component(env_name)
    training = training_component(env_name)

    @pipeline(
        name="rister_ml_pipeline",
        display_name="Rister ML Pipeline",
        description="Rister (vers) + WerkExpert (Blob) → merge → LightGBM training → MLflow",
    )
    def rister_pipeline():
        stap_rister = rister()
        stap_merge = merge(
            rister_data=stap_rister.outputs.processed,
            # WerkExpert CSV komt uit Blob Storage (handmatig geupload vanaf lokale Mac)
            werkexpert_data=Input(type="uri_file", path=WERKEXPERT_BLOB_URI),
        )
        stap_training = training(
            merged_data=stap_merge.outputs.merged,
        )
        return {"models": stap_training.outputs.models}

    return rister_pipeline()


# ─── Submit ────────────────────────────────────────────────────────────────────

def submit():
    ml_client = get_ml_client()

    print("Environment registreren...")
    env = get_or_create_environment(ml_client)
    env_name = f"{env.name}:{env.version}"

    print("Pipeline bouwen...")
    pipeline_job = build_pipeline(env_name)
    pipeline_job.settings.default_compute = "rister-compute"

    print("Pipeline submitten...")
    job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="rister-lightgbm-v1-plushoeveelheid")
    print(f"Pipeline gestart: {job.studio_url}")
    return job


if __name__ == "__main__":
    submit()
