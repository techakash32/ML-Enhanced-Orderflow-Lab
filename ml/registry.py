# ml/registry.py
"""MLflow tracking + model-registry helpers (SQLite backend, no external services).
The registry holds the `champion` alias; a local copy of the champion bundle is kept
in ML_MODEL_PATH so the live container can run without MLflow installed/reachable."""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
CHAMPION_ALIAS = "champion"
DEFAULT_URI = "sqlite:///logs/ml/mlruns/mlflow.db"
DEFAULT_ARTIFACTS = "logs/ml/mlruns/artifacts"


def resolve_uri(uri: str | None) -> str:
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_URI
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        os.makedirs(os.path.dirname(uri[len("sqlite:///"):]) or ".", exist_ok=True)
    return uri


def setup(uri: str | None, experiment: str):
    import mlflow
    uri = resolve_uri(uri)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(experiment) is None:
        art = os.path.abspath(DEFAULT_ARTIFACTS) if uri.startswith("sqlite") else None
        if art:
            os.makedirs(art, exist_ok=True)
        client.create_experiment(experiment, artifact_location=("file://" + art) if art else None)
    mlflow.set_experiment(experiment)
    return mlflow, client


def champion(client, name: str):
    try:
        return client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except Exception:
        return None


def register(mlflow, client, run_id: str, name: str, artifact_path: str = "bundle"):
    try:
        client.create_registered_model(name)
    except Exception:
        pass                                              # already exists
    mv = client.create_model_version(name=name, source=f"runs:/{run_id}/{artifact_path}", run_id=run_id)
    return mv


def promote(client, name: str, version) -> None:
    client.set_registered_model_alias(name, CHAMPION_ALIAS, str(version))


def download_champion(tracking_uri: str, name: str, dst: str) -> tuple[str, str] | None:
    """Fetch the registry champion bundle into dst. Returns (dst, version) or None."""
    import mlflow
    mlflow.set_tracking_uri(resolve_uri(tracking_uri))
    client = mlflow.MlflowClient()
    mv = champion(client, name)
    if mv is None:
        return None
    os.makedirs(dst, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=dst)
    return path, str(mv.version)
