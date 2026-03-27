"""
Tests voor src/config.py

Controleert dat omgevingsvariabelen correct worden opgepikt
en dat de defaults kloppen.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_data_dir_override(monkeypatch, tmp_path):
    """DATA_DIR omgevingsvariabele wordt correct overgenomen."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # config opnieuw laden om env-var op te pikken
    import importlib

    import config

    importlib.reload(config)
    assert config.DATA_DIR == tmp_path
    assert config.PROCESSED_DIR == tmp_path / "processed"


def test_data_dir_default():
    """Zonder DATA_DIR omgevingsvariabele valt config terug op PROJECT_ROOT/data."""
    import importlib

    import config

    importlib.reload(config)
    assert config.DATA_DIR.name == "data"
    assert config.PROCESSED_DIR == config.DATA_DIR / "processed"


def test_models_dir_override(monkeypatch, tmp_path):
    """MODELS_DIR omgevingsvariabele wordt correct overgenomen."""
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "mymodels"))
    import importlib

    import config

    importlib.reload(config)
    assert config.MODELS_DIR == tmp_path / "mymodels"
    assert config.MODELS_ONNX_DIR == tmp_path / "mymodels" / "onnx"


def test_mlflow_uri_fallback_chain(monkeypatch):
    """MLFLOW_TRACKING_URI heeft voorrang op MLFLOW_URI."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_URI", "http://custom:5001")
    import importlib

    import config

    importlib.reload(config)
    assert config.MLFLOW_URI == "http://custom:5001"


def test_mlflow_tracking_uri_prioriteit(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://azureml/tracking")
    monkeypatch.setenv("MLFLOW_URI", "http://local:5001")
    import importlib

    import config

    importlib.reload(config)
    assert config.MLFLOW_URI == "http://azureml/tracking"


def test_aml_defaults():
    """Standaard AML-waarden kloppen met de verwachte resourcenamen."""
    import importlib

    import config

    importlib.reload(config)
    assert config.AML_RESOURCE_GROUP == "rister-ml"
    assert config.AML_WORKSPACE == "rister-aml"
