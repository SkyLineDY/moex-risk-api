import sqlite3
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from moex_api.api import create_app
from moex_api.model import calculate_risk, classify
from moex_api.schemas import EXAMPLE, StockRequest


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3")) as client:
        yield client


def test_health_model_and_contract(client):
    assert client.get("/health").json() == {"status": "ok", "model_ready": True, "storage": "sqlite"}
    assert client.get("/model-info").json()["model_type"] == "deterministic_heuristic"
    assert len(client.get("/instruments").json()) == 5
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/model-info", "/instruments", "/predict", "/predictions", "/predictions/{request_id}"}
    assert {"201", "400", "422", "503"} <= set(schema["paths"]["/predict"]["post"]["responses"])


def test_create_and_retrieve(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 201
    result = response.json()
    assert result["risk_score"] == pytest.approx(0.185)
    assert result["risk_level"] == "low"
    assert sum(f["contribution"] for f in result["factors"]) == pytest.approx(result["risk_score"])
    assert result["input_data"] == EXAMPLE
    assert response.headers["x-request-id"]
    assert float(response.headers["x-process-time-ms"]) >= 0
    assert client.get(response.headers["location"]).json() == result


@pytest.mark.parametrize("patch", [
    {"annual_volatility_pct": -1}, {"annual_volatility_pct": 301},
    {"max_drawdown_pct": 101}, {"avg_daily_turnover_mrub": -1},
    {"bid_ask_spread_pct": -0.1}, {"observation_days": 19},
    {"observation_days": 253}, {"observation_days": 60.5},
    {"observation_days": True}, {"unexpected": 1}, {"ticker": "!!!"},
    {"annual_volatility_pct": "NaN"}, {"bid_ask_spread_pct": "Infinity"},
])
def test_invalid_input(client, patch):
    response = client.post("/predict", json=EXAMPLE | patch)
    assert response.status_code == 422
    assert client.get("/predictions").json() == []


def test_missing_field(client):
    payload = EXAMPLE.copy()
    del payload["max_drawdown_pct"]
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("patch", [{"ticker": "UNKNOWN"}, {"board": "TEST"}])
def test_business_error(client, patch):
    assert client.post("/predict", json=EXAMPLE | patch).status_code == 400
    assert client.get("/predictions").json() == []


def test_normalization(client):
    result = client.post("/predict", json=EXAMPLE | {"ticker": " sber ", "board": "tqbr"})
    assert result.status_code == 201
    assert result.json()["ticker"] == "SBER"


def test_missing_id(client):
    assert client.get(f"/predictions/{uuid4()}").status_code == 404
    assert client.get("/predictions/not-a-uuid").status_code == 422


@pytest.mark.parametrize("query", ["limit=-5", "limit=0", "limit=101", "offset=-1", "risk_level=bad"])
def test_invalid_queries(client, query):
    assert client.get("/predictions?" + query).status_code == 422


def test_filter_order_and_pagination(client):
    first = client.post("/predict", json=EXAMPLE).json()
    high_data = EXAMPLE | {"ticker": "MOEX", "annual_volatility_pct": 80,
                          "max_drawdown_pct": 60, "avg_daily_turnover_mrub": 0, "bid_ask_spread_pct": 2}
    second = client.post("/predict", json=high_data).json()
    third = client.post("/predict", json=EXAMPLE).json()
    assert client.get("/predictions?limit=2").json() == [third, second]
    assert client.get("/predictions?limit=1&offset=2").json() == [first]
    assert client.get("/predictions?risk_level=high&ticker=moex").json() == [second]
    assert client.get("/predictions?risk_level=low&limit=1&offset=1").json() == [first]
    assert client.get("/predictions?ticker=UNKNOWN").json() == []


@pytest.mark.parametrize("score,level", [(0,"low"),(.349999,"low"),(.35,"medium"),(.649999,"medium"),(.65,"high"),(1,"high")])
def test_thresholds(score, level):
    assert classify(score).value == level


def test_model_extremes_and_monotonicity():
    low = EXAMPLE | {"annual_volatility_pct": 0, "max_drawdown_pct": 0,
                     "avg_daily_turnover_mrub": 100, "bid_ask_spread_pct": 0}
    high = low | {"annual_volatility_pct": 300, "max_drawdown_pct": 100,
                  "avg_daily_turnover_mrub": 0, "bid_ask_spread_pct": 100}
    assert calculate_risk(StockRequest(**low))[0] == 0
    assert calculate_risk(StockRequest(**high))[0] == 1
    baseline = calculate_risk(StockRequest(**EXAMPLE))[0]
    for patch in [{"annual_volatility_pct": 60}, {"max_drawdown_pct": 50},
                  {"avg_daily_turnover_mrub": 1}, {"bid_ask_spread_pct": 1}]:
        assert calculate_risk(StockRequest(**(EXAMPLE | patch)))[0] > baseline


def test_persistence(tmp_path):
    path = tmp_path / "persistent.sqlite3"
    with TestClient(create_app(path)) as first:
        response = first.post("/predict", json=EXAMPLE)
        result, location = response.json(), response.headers["location"]
    with TestClient(create_app(path)) as second:
        assert second.get(location).json() == result


def test_unavailable_model(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3", model_ready=False)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=EXAMPLE).status_code == 503
        assert client.get("/model-info").json()["ready"] is False
        assert client.get("/predictions").json() == []


def test_storage_failure(client, monkeypatch):
    def fail(*args):
        raise sqlite3.OperationalError("private database path")
    monkeypatch.setattr(client.app.state.storage, "save", fail)
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 503
    assert "private" not in response.text
    assert client.get("/predictions").json() == []
