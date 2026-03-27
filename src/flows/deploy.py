"""
Deployt de Rister ML Pipeline als een Prefect-deployment met weekschema.

Eenmalig uitvoeren om de deployment aan te maken:
    python src/flows/deploy.py

Daarna draait de pipeline automatisch elke maandag om 02:00.
Schema aanpassen: wijzig de cron-string hieronder.

Prefect UI starten om status te bekijken:
    prefect server start   →   http://127.0.0.1:4200

Worker starten (nodig om deployments uit te voeren):
    prefect worker start --pool rister-pool
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefect.client.schemas.schedules import CronSchedule

from flows.pipeline_flow import rister_pipeline

if __name__ == "__main__":
    rister_pipeline.serve(
        name="rister-wekelijks",
        schedules=[
            CronSchedule(
                cron="0 2 1 1,4,7,10 *",   # elk kwartaal: 1 jan, 1 apr, 1 jul, 1 okt om 02:00
                timezone="Europe/Amsterdam",
            )
        ],
        parameters={
            "skip_rister": False,
            "skip_werkexpert": False,
            "skip_merge": False,
            "skip_train": False,
        },
        tags=["rister", "ml", "wekelijks"],
    )
