"""Azure ML pipeline definitie voor Rister scheduled training.

Stappen:
  1. Rister dataprep (PostgreSQL → rister.csv)
  2. Merge (rister.csv + werkexpert.parquet → dataframe_gecombineerd.csv)
  3. Training (LightGBM regressor + ranker → ONNX modellen)

Wordt aangeroepen via: python src/aml/pipeline.py
"""

import os

from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from src.config import AML_RESOURCE_GROUP, AML_SUBSCRIPTION_ID, AML_WORKSPACE


def get_ml_client() -> MLClient:
    """Authenticeer en return Azure ML workspace client."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=AML_SUBSCRIPTION_ID,
        resource_group_name=AML_RESOURCE_GROUP,
        workspace_name=AML_WORKSPACE,
    )


def get_environment(ml_client: MLClient) -> Environment:
    """Maak of haal de training environment op."""
    env = Environment(
        name="rister-training-env",
        description="Python 3.11 met LightGBM, MLflow, pandas, geopandas",
        conda_file={
            "name": "rister-env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.11",
                "pip",
                {
                    "pip": [
                        "pandas",
                        "numpy",
                        "scikit-learn",
                        "lightgbm",
                        "optuna",
                        "mlflow",
                        "azureml-mlflow",
                        "onnxmltools",
                        "onnxruntime",
                        "skl2onnx",
                        "sqlalchemy",
                        "psycopg2-binary",
                        "geopandas",
                        "pyarrow",
                    ]
                },
            ],
        },
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )
    return env


@dsl.pipeline(
    name="rister-training-pipeline",
    description="Scheduled training pipeline: dataprep → merge → train",
)
def rister_pipeline(
    postgres_uri: str,
    werkexpert_data: Input,
    force_tuning: bool = False,
):
    """Definieer de 3-staps Azure ML pipeline."""

    env = "rister-training-env@latest"

    # Stap 1: Rister dataprep
    rister_step = command(
        name="rister_dataprep",
        display_name="Rister DataPrep",
        command=(
            "python -m src.dataprep.rister"
            " --postgres-uri ${{inputs.postgres_uri}}"
            " --output-dir ${{outputs.output_dir}}"
        ),
        inputs={"postgres_uri": postgres_uri},
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=env,
        code=".",
    )

    # Stap 2: Merge
    merge_step = command(
        name="merge_dataprep",
        display_name="Merge Rister + WerkExpert",
        command=(
            "python -m src.dataprep.merge"
            " --rister-path ${{inputs.rister_dir}}/rister.csv"
            " --werkexpert-path ${{inputs.werkexpert_data}}"
            " --output-dir ${{outputs.output_dir}}"
        ),
        inputs={
            "rister_dir": rister_step.outputs.output_dir,
            "werkexpert_data": werkexpert_data,
        },
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=env,
        code=".",
    )

    # Stap 3: Training
    force_flag = " --force-tuning" if force_tuning else ""
    train_step = command(
        name="train_lightgbm",
        display_name="Train LightGBM",
        command=(
            "python -m src.train.train_lightgbm"
            " --input-dir ${{inputs.input_dir}}"
            " --output-dir ${{outputs.output_dir}}" + force_flag
        ),
        inputs={"input_dir": merge_step.outputs.output_dir},
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=env,
        code=".",
    )

    return {"model_output": train_step.outputs.output_dir}


def submit_pipeline(ml_client: MLClient, force_tuning: bool = False):
    """Submit de pipeline naar Azure ML."""
    # Haal PostgreSQL URI op (via Key Vault of env var)
    postgres_uri = os.environ.get("POSTGRES_URI", "")
    if not postgres_uri:
        raise ValueError("POSTGRES_URI environment variable is niet gezet")

    # WerkExpert data asset (handmatig geupload naar Blob)
    werkexpert_input = Input(
        type=AssetTypes.URI_FILE,
        path="azureml://datastores/workspaceblobstore/paths/rister-data/werkexpert.parquet",
    )

    pipeline_job = rister_pipeline(
        postgres_uri=postgres_uri,
        werkexpert_data=werkexpert_input,
        force_tuning=force_tuning,
    )

    pipeline_job.settings.default_compute = "rister-cluster"

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Pipeline submitted: {submitted.name}")
    print(f"Studio URL: {submitted.studio_url}")
    return submitted


def main():
    """Entry point voor cd.yml en handmatige runs."""
    ml_client = get_ml_client()

    # Registreer environment
    env = get_environment(ml_client)
    ml_client.environments.create_or_update(env)
    print(f"Environment geregistreerd: {env.name}")

    # Submit pipeline
    force = os.environ.get("FORCE_TUNING", "false").lower() == "true"
    submit_pipeline(ml_client, force_tuning=force)


if __name__ == "__main__":
    main()
