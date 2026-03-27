"""
ML Pipeline: dataprep → merge → training
Runt alle stappen in volgorde. Elke stap kan ook los uitgevoerd worden.

Gebruik:
    python src/pipeline.py                    # alle stappen
    python src/pipeline.py --skip-rister      # sla Rister dataprep over
    python src/pipeline.py --only-train       # alleen training
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _stap(naam, fn):
    log.info(f"=== Start: {naam} ===")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    log.info(f"=== Klaar: {naam} ({elapsed:.0f}s) ===")


def main():
    parser = argparse.ArgumentParser(description="Rister ML Pipeline")
    parser.add_argument("--skip-rister", action="store_true", help="Sla Rister dataprep over")
    parser.add_argument("--skip-werkexpert", action="store_true", help="Sla WerkExpert dataprep over")
    parser.add_argument("--skip-merge", action="store_true", help="Sla merge stap over")
    parser.add_argument("--skip-train", action="store_true", help="Sla training over")
    parser.add_argument("--only-train", action="store_true", help="Alleen training uitvoeren")
    args = parser.parse_args()

    t_start = time.time()

    if not args.only_train:
        if not args.skip_rister:
            from dataprep.rister import run as run_rister
            _stap("Rister dataprep", run_rister)

        if not args.skip_werkexpert:
            from dataprep.werkexpert import run as run_werkexpert
            _stap("WerkExpert dataprep", run_werkexpert)

        if not args.skip_merge:
            from dataprep.merge import run as run_merge
            _stap("Merge", run_merge)

    if not args.skip_train:
        from train.train_lightgbm import run as run_train
        _stap("LightGBM training", run_train)

    log.info(f"Pipeline klaar in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
