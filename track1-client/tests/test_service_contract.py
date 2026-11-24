import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.settings import ServiceSettings, validate_settings_or_raise


def test_healthz_vs_readyz_when_unready():
    cfg = ServiceSettings(app_name="test-svc", model_id="fake-7b", force_unready=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        h = client.get("/healthz")
        assert h.status_code == 200
        assert h.json()["status"] == "alive"

        r = client.get("/readyz")
        assert r.status_code == 503
        assert r.json()["error_code"] == "MODEL_NOT_READY"

        g = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 8})
        assert g.status_code == 503
        assert g.json()["error_code"] == "SERVICE_UNAVAILABLE"


def test_valid_generate_and_request_id_propagation():
    cfg = ServiceSettings(app_name="test-svc", model_id="fake-7b", max_prompt_chars=100, max_new_tokens=64)
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

        res = client.post(
            "/generate",
            headers={"X-Request-ID": "req-custom-50"},
            json={"prompt": "Explain KV cache", "max_new_tokens": 16},
        )
        assert res.status_code == 200
        assert res.headers["X-Request-ID"] == "req-custom-50"
        body = res.json()
        assert body["request_id"] == "req-custom-50"
        assert body["model_id"] == "fake-7b"
        assert body["tokens_generated"] >= 1
        assert body["latency_ms"] >= 0.0


def test_invalid_inputs_return_structured_errors():
    cfg = ServiceSettings(max_prompt_chars=10, max_new_tokens=16)
    app = create_app(cfg)
    with TestClient(app) as client:
        empty_res = client.post("/generate", json={"prompt": "", "max_new_tokens": 8})
        assert empty_res.status_code == 422
        assert empty_res.json()["error_code"] == "VALIDATION_ERROR"

        long_res = client.post("/generate", json={"prompt": "This prompt is way too long", "max_new_tokens": 8})
        assert long_res.status_code == 400
        assert long_res.json()["error_code"] == "PROMPT_TOO_LONG"

        tok_res = client.post("/generate", json={"prompt": "Short", "max_new_tokens": 128})
        assert tok_res.status_code == 400
        assert tok_res.json()["error_code"] == "MAX_TOKENS_EXCEEDED"


def test_invalid_env_settings_fail_fast(monkeypatch):
    monkeypatch.setenv("INFERENCE_MAX_NEW_TOKENS", "-10")
    with pytest.raises(RuntimeError, match="Invalid service configuration"):
        validate_settings_or_raise()


def test_graceful_shutdown_resets_readiness():
    app = create_app(ServiceSettings())
    with TestClient(app) as client:
        assert app.state.is_ready is True
        assert client.get("/readyz").status_code == 200
    assert app.state.is_ready is False
    assert app.state.in_flight == 0
