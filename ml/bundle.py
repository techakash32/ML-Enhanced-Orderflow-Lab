# ml/bundle.py
"""A model bundle is a directory: model.txt (native LightGBM) + manifest.json.
Everything the live gate needs - feature order, classes, barrier geometry,
training date - travels with the model. No pickle, so it loads anywhere."""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import lightgbm as lgb

from features.orderflow import MODEL_FEATURES


def schema_hash(names: list[str]) -> str:
    return hashlib.sha1(",".join(names).encode()).hexdigest()[:12]


def save_bundle(booster: lgb.Booster, manifest: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "model.txt"))
    manifest = {**manifest, "feature_schema_hash": schema_hash(manifest["feature_names"])}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out_dir


def load_bundle(path: str):
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    booster = lgb.Booster(model_file=os.path.join(path, "model.txt"))
    return booster, manifest


def predict_favorable(booster: lgb.Booster, manifest: dict, X):
    """P(target hit first) for each row of X (a DataFrame with the manifest's features)."""
    p = booster.predict(X[manifest["feature_names"]])
    return p[:, manifest["favorable_index"]] if p.ndim == 2 else p


def copy_bundle_atomic(src: str, dst: str) -> None:
    """Replace dst with src; manifest is written last so readers never see a half-copy."""
    tmp = dst.rstrip("/\\") + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)


def compatible(manifest: dict) -> bool:
    return manifest.get("feature_names") == MODEL_FEATURES
