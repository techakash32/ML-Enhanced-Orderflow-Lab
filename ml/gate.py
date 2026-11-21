# ml/gate.py
"""Live inference wrapper used by engine/decision.py (Phase 4)."""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from config import settings
from features.orderflow import to_model_frame
from ml.bundle import compatible, load_bundle, predict_favorable

log = logging.getLogger(__name__)


class MLGate:
    """Loads the champion model and returns P(favourable) for (features, side).

    source: local    -> settings.ML_MODEL_PATH
            registry -> MLflow registry alias `champion`
            auto     -> registry if MLFLOW_TRACKING_URI is set and has a champion,
                        otherwise the local path
    The model is re-checked every ML_RELOAD_CHECK_S so a weekly retrain is picked up
    without restarting the paper trader.
    """

    def __init__(self, path: str | None = None, source: str | None = None,
                 tracking_uri: str | None = None, name: str | None = None,
                 reload_check_s: float | None = None):
        self.path = path or settings.ML_MODEL_PATH
        self.source = (source or settings.ML_MODEL_SOURCE)
        self.uri = settings.MLFLOW_TRACKING_URI if tracking_uri is None else tracking_uri
        self.name = name or settings.ML_REGISTERED_MODEL
        self.reload_check_s = settings.ML_RELOAD_CHECK_S if reload_check_s is None else reload_check_s
        self.booster = None
        self.manifest: dict = {}
        self.version = None
        self._stamp = None
        self._last_check = 0.0
        self._warned = False
        self.load()

    @property
    def available(self) -> bool:
        return self.booster is not None

    # ---- loading ----
    def _local_stamp(self):
        m = os.path.join(self.path, "manifest.json")
        return os.path.getmtime(m) if os.path.exists(m) else None

    def load(self) -> bool:
        self._last_check = time.time()
        path, ver = None, None
        if self.source in ("registry", "auto") and self.uri:
            try:
                from ml.registry import download_champion
                got = download_champion(self.uri, self.name, os.path.join(self.path + "_registry_cache"))
                if got:
                    path, ver = got
            except Exception as exc:
                log.warning("ML registry load failed (%s)", exc)
        if path is None and self.source in ("local", "auto", "registry"):
            if self._local_stamp() is not None:
                path, ver = self.path, "local"
        if path is None:
            self._on_missing()
            return False
        try:
            booster, manifest = load_bundle(path)
        except Exception as exc:
            log.error("ML model at %s failed to load: %s", path, exc)
            return False
        if not compatible(manifest):
            log.error("ML model feature schema %s != code schema - refusing to use it",
                      manifest.get("feature_schema_hash"))
            return False
        self.booster, self.manifest, self.version = booster, manifest, ver
        self._stamp = self._local_stamp() if ver == "local" else ver
        log.info("ML gate loaded model %s (%s, trained %s, objective=%s)",
                 manifest.get("model_version"), ver, manifest.get("created_utc"),
                 manifest.get("objective"))
        return True

    def _on_missing(self) -> None:
        if not self._warned:
            log.warning("ML gate: no model found at %s (source=%s) - falling back to rule-based gate",
                        self.path, self.source)
            self._warned = True

    def _registry_version(self):
        from ml.registry import champion, resolve_uri
        import mlflow
        mlflow.set_tracking_uri(resolve_uri(self.uri))
        mv = champion(mlflow.MlflowClient(), self.name)
        return str(mv.version) if mv else None

    def maybe_reload(self) -> None:
        """Cheap periodic check (every reload_check_s): pick up a newly promoted champion."""
        if time.time() - self._last_check < self.reload_check_s:
            return
        self._last_check = time.time()
        try:
            if self.version == "local":
                if self._local_stamp() != self._stamp:
                    self.load()
            elif self.version is not None:                      # registry-sourced
                if self._registry_version() != self.version:
                    self.load()
            else:                                               # nothing loaded yet
                self.load()
        except Exception as exc:
            log.warning("ML reload check failed: %s", exc)

    # ---- inference ----
    def p_favorable(self, feats: dict, side: int, atr_pct: float) -> float:
        X = to_model_frame(feats, side, atr_pct)
        return float(predict_favorable(self.booster, self.manifest, X)[0])

    def status(self) -> dict:
        return {"available": self.available, "version": self.version,
                "model_version": self.manifest.get("model_version"),
                "trained": self.manifest.get("created_utc")}
