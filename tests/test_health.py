import json

from fastapi.testclient import TestClient

from config import settings


def test_health_endpoint(tmp_path, monkeypatch):
    hb = tmp_path / "hb.json"
    import time
    hb.write_text(json.dumps({"ts": time.time(), "stats": {"trades": 5, "books": 7, "reconnects": 0},
                              "last_snapshot_ms": {"BTCUSDT": int(time.time() * 1000) - 5000}, "resets": {}}))
    monkeypatch.setattr(settings, "FEATURE_HEARTBEAT_PATH", str(hb))
    monkeypatch.setattr(settings, "ML_MODEL_PATH", str(tmp_path / "models" / "champion"))
    monkeypatch.setattr(settings, "USE_ML_GATE", True)
    from scripts.dashboard import app
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["recorder"]["trades_seen"] == 5 and j["model"]["present"] is False
    monkeypatch.setattr(settings, "FEATURE_HEARTBEAT_PATH", str(tmp_path / "missing.json"))
    assert TestClient(app).get("/health").json()["status"] == "degraded"
