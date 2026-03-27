"""
Azure ML deployment: pipeline met kwartaalschema
Maakt een compute cluster aan (als die nog niet bestaat) en
registreert de pipeline als scheduled job.

Eenmalig uitvoeren na het aanmaken van de Azure ML Workspace:
    python src/aml/deploy.py

Daarna draait de pipeline automatisch elk kwartaal op de 1e van de maand om 02:00 (Amsterdam):
januari, april, juli, oktober.
Status bekijken: https://ml.azure.com
"""

import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    JobSchedule,
    RecurrencePattern,
    RecurrenceTrigger,
)
from azure.identity import DefaultAzureCredential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aml.pipeline import build_pipeline, get_ml_client, get_or_create_environment
from config import AML_RESOURCE_GROUP, AML_SUBSCRIPTION_ID, AML_WORKSPACE

COMPUTE_NAME = "rister-compute"
SCHEDULE_NAME = "rister-kwartaal"


def ensure_compute(ml_client: MLClient):
    """Maak compute cluster aan als die nog niet bestaat."""
    try:
        ml_client.compute.get(COMPUTE_NAME)
        print(f"Compute cluster '{COMPUTE_NAME}' bestaat al")
    except Exception:
        print(f"Compute cluster '{COMPUTE_NAME}' aanmaken...")
        ml_client.compute.begin_create_or_update(
            AmlCompute(
                name=COMPUTE_NAME,
                size="Standard_DS2_v2",  # 2 vCPU, 7 GB RAM — voldoende voor 28k rijen
                min_instances=0,  # schaalt naar 0 als idle (geen kosten)
                max_instances=1,
                idle_time_before_scale_down=60,  # sneller afschalen dan standaard
                tier="LowPriority",  # spot instance: ~80% goedkoper
            )
        ).result()
        print("Compute cluster aangemaakt")


def deploy():
    ml_client = get_ml_client()

    # Compute cluster
    ensure_compute(ml_client)

    # Environment
    print("Environment registreren...")
    env = get_or_create_environment(ml_client)
    env_name = f"{env.name}:{env.version}"

    # Pipeline bouwen
    print("Pipeline bouwen...")
    pipeline_job = build_pipeline(env_name)
    pipeline_job.settings.default_compute = COMPUTE_NAME

    # Schedule aanmaken: elk kwartaal op de 1e van de maand om 02:00 Amsterdam
    # Maanden: januari (1), april (4), juli (7), oktober (10)
    print(f"Schedule '{SCHEDULE_NAME}' aanmaken...")
    schedule = JobSchedule(
        name=SCHEDULE_NAME,
        trigger=RecurrenceTrigger(
            frequency="month",
            interval=3,
            schedule=RecurrencePattern(
                hours=[2],
                minutes=[0],
                month_days=[1],
            ),
            time_zone="W. Europe Standard Time",
        ),
        create_job=pipeline_job,
    )

    ml_client.schedules.begin_create_or_update(schedule).result()
    print("Schedule aangemaakt: elk kwartaal op de 1e om 02:00")
    print(f"Bekijk via: https://ml.azure.com/experiments (workspace: {AML_WORKSPACE})")


if __name__ == "__main__":
    deploy()
