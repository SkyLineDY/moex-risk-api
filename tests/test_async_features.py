import asyncio
import json
import sqlite3
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from moex_api.api import create_app
from moex_api.batching import DynamicBatcher
from moex_api.config import Settings
from moex_api.schemas import EXAMPLE, StockRequest
from moex_api.services import InferenceModel, make_prediction


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3")) as client:
        yield client


def test_trace_in_response_errors_and_logs(client, caplog):
    with caplog.at_level("INFO", logger="moex_api"):
        result = client.post("/predict", json=EXAMPLE, headers={"X-Request-ID": "client-42"})
    assert result.headers["x-request-id"] == "client-42"
    assert result.json()["trace_id"] == "client-42"
    assert result.json()["request_id"] == "client-42"
    assert "trace=client-42" in caplog.text and "stage=saved" in caplog.text
    invalid = client.post("/predict", json={}, headers={"X-Request-ID": "invalid-42"})
    assert invalid.status_code == 422
    assert invalid.json()["request_id"] == invalid.headers["x-request-id"] == "invalid-42"
    assert "input" not in invalid.json()["detail"][0]


def test_bad_trace(client):
    response = client.get("/health", headers={"X-Request-ID": "x"*129})
    assert response.status_code == 400
    assert len(response.headers["x-request-id"]) == 36


def test_explicit_batch(client):
    response = client.post("/batch-predict", json={"items": [EXAMPLE, EXAMPLE | {"ticker":"MOEX"}]}, headers={"X-Request-ID":"batch-42"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == response.json()["batch_size"] == 2
    assert items[0]["ticker"] == "SBER" and items[1]["ticker"] == "MOEX"
    assert len({p["prediction_id"] for p in items}) == 2
    assert {p["trace_id"] for p in items} == {"batch-42"}
    assert {p["source"] for p in items} == {"batch"}
    assert client.get("/requests/batch-42/predictions").json() == items
    assert client.get("/predictions/batch-42").status_code == 409


@pytest.mark.parametrize("items", [[], [EXAMPLE]*9, [EXAMPLE, EXAMPLE | {"ticker":"UNKNOWN"}], [EXAMPLE, EXAMPLE | {"max_drawdown_pct":-1}]])
def test_bad_batch_does_not_save(client, items):
    assert client.post("/batch-predict",json={"items":items}).status_code in {400,422}
    assert client.get("/predictions").json() == []


def test_batch_atomicity(client, monkeypatch):
    async def duplicate(items, traces, source):
        result = make_prediction(items[0], traces[0], source)
        return [result,result]
    monkeypatch.setattr(client.app.state.batcher.model,"predict_many",duplicate)
    assert client.post("/batch-predict",json={"items":[EXAMPLE,EXAMPLE]}).status_code == 503
    assert client.get("/predictions").json() == []


def test_weekday_pagination_and_empty(client):
    trace = "pages"
    for _ in range(3):
        client.post("/predict",json=EXAMPLE,headers={"X-Request-ID":trace})
    weekday = datetime.now(timezone.utc).strftime("%A").lower()
    path = f"/predictions/by-weekday?weekday={weekday}&per_page=2"
    result = client.get(path).json()
    assert len(result["data"]) == 2
    assert result["meta"]["total_items"] == 3
    assert result["meta"]["total_pages"] == 2
    assert result["meta"]["timezone"] == "UTC"
    second = client.get(result["meta"]["next_page"]).json()
    assert len(second["data"]) == 1 and second["meta"]["prev_page"]
    assert client.get(path+"&page=3").status_code == 404
    other = "sunday" if weekday != "sunday" else "monday"
    empty = client.get(f"/predictions/by-weekday?weekday={other}").json()
    assert empty["data"] == [] and empty["meta"]["total_pages"] == 0
    assert client.get(f"/predictions/by-weekday?weekday={other}&page=2").status_code == 404


@pytest.mark.parametrize("query,expected", [("weekday=bad",400),("weekday=monday&page=0",422),("weekday=monday&per_page=101",422)])
def test_bad_pagination(client,query,expected):
    assert client.get("/predictions/by-weekday?"+query).status_code == expected


def test_parallel_queue_mapping(tmp_path):
    async def scenario():
        app = create_app(tmp_path / "queue.sqlite3",settings=Settings(batch_wait_ms=100))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                results = await asyncio.gather(*[client.post("/predict-queued",json=EXAMPLE | {"annual_volatility_pct":n*4},headers={"X-Request-ID":f"queue-{n}"}) for n in range(12)])
                assert all(r.status_code == 200 for r in results)
                for n,r in enumerate(results):
                    assert r.json()["trace_id"] == f"queue-{n}"
                    assert r.json()["input_data"]["annual_volatility_pct"] == n*4
                status = (await client.get("/queue-status")).json()
                assert 1 < status["max_observed_batch_size"] <= 8
                assert status["processed_items"] == 12
                assert status["processed_batches"] < 12
    asyncio.run(scenario())


def test_queue_full_timeout_and_recovery():
    async def scenario():
        settings = Settings(queue_size=1,max_batch_size=1,batch_wait_ms=0,inference_timeout=.02)
        model = InferenceModel(settings)
        gate = asyncio.Event()
        async def slow(*args):
            await gate.wait()
            return []
        model.predict_many = slow
        batcher = DynamicBatcher(model,settings)
        batcher.start()
        first=asyncio.create_task(batcher.submit(StockRequest(**EXAMPLE),"one"))
        while batcher.queue.qsize():
            await asyncio.sleep(0)
        await asyncio.sleep(.002)
        second=asyncio.create_task(batcher.submit(StockRequest(**EXAMPLE),"two"))
        await asyncio.sleep(.002)
        with pytest.raises(HTTPException) as full:
            await batcher.submit(StockRequest(**EXAMPLE),"three")
        assert full.value.status_code == 429 and full.value.headers["Retry-After"] == "1"
        results=await asyncio.gather(first,second,return_exceptions=True)
        assert [r.status_code for r in results] == [504,504]
        gate.set()
        await asyncio.wait_for(batcher.queue.join(),1)
        model.predict_many = InferenceModel(settings).predict_many
        await batcher.close()
        assert batcher.queue.empty()
    asyncio.run(scenario())


def test_timeout_does_not_persist(tmp_path):
    settings=Settings(inference_timeout=.001,batch_wait_ms=30)
    with TestClient(create_app(tmp_path/'timeout.sqlite3',settings=settings)) as client:
        assert client.post('/predict-queued',json=EXAMPLE).status_code == 504
        assert client.get('/predictions').json() == []


def test_worker_recovers_from_error():
    async def scenario():
        settings=Settings(batch_wait_ms=0)
        model=InferenceModel(settings)
        original=model.predict_many
        async def fail(*args):
            raise RuntimeError('model failed')
        model.predict_many=fail
        batcher=DynamicBatcher(model,settings)
        batcher.start()
        with pytest.raises(HTTPException) as exc:
            await batcher.submit(StockRequest(**EXAMPLE),'failed')
        assert exc.value.status_code == 503
        model.predict_many=original
        assert (await batcher.submit(StockRequest(**EXAMPLE),'recovered')).trace_id == 'recovered'
        await batcher.close()
    asyncio.run(scenario())


def mock_llm(request):
    body=json.loads(request.content)
    assert request.url.path.endswith('/chat/completions')
    assert body['model'] == 'auto'
    return httpx.Response(200,json={'model':'test-model','choices':[{'message':{'content':'Вклад волатильности 0,12, просадки 0,06, спреда 0,005. Балл 0,185.'},'finish_reason':'stop'}],'usage':{'total_tokens':30}})


def test_llm_and_combined_response(tmp_path):
    with TestClient(create_app(tmp_path/'llm.sqlite3',llm_transport=httpx.MockTransport(mock_llm))) as client:
        chat=client.post('/llm/chat',json={'prompt':'Что такое волатильность?'},headers={'X-Request-ID':'llm-1'})
        assert chat.status_code == 200 and chat.json()['returned_model'] == 'test-model'
        result=client.post('/predict-with-llm',json=EXAMPLE)
        assert result.status_code == 200
        assert result.json()['prediction']['risk_score'] == .185
        assert result.json()['llm']['request_id'] == result.headers['x-request-id']
        assert len(client.get('/predictions').json()) == 1


@pytest.mark.parametrize('mode,expected',[('timeout',504),('http',502),('json',502),('empty',502)])
def test_llm_failures_do_not_save(tmp_path,mode,expected):
    def upstream(request):
        if mode == 'timeout': raise httpx.ReadTimeout('timeout')
        if mode == 'http': return httpx.Response(429,text='quota')
        if mode == 'json': return httpx.Response(200,text='broken json')
        return httpx.Response(200,json={'choices':[{'message':{'content':''}}]})
    with TestClient(create_app(tmp_path/'llm.sqlite3',llm_transport=httpx.MockTransport(upstream))) as client:
        response=client.post('/predict-with-llm',json=EXAMPLE,headers={'X-Request-ID':'upstream'})
        assert response.status_code == expected
        assert response.json()['request_id'] == 'upstream'
        assert client.get('/predictions').json() == []


def test_llm_blank_prompt(client):
    assert client.post('/llm/chat',json={'prompt':'   '}).status_code == 422


def test_legacy_database(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    prediction=make_prediction(StockRequest(**EXAMPLE),'old','single').model_dump(mode='json')
    prediction['request_id'] = prediction.pop('prediction_id')
    for key in ['trace_id','source']: prediction.pop(key)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE predictions (seq INTEGER PRIMARY KEY AUTOINCREMENT,request_id TEXT NOT NULL UNIQUE,ticker TEXT NOT NULL,risk_level TEXT NOT NULL,payload TEXT NOT NULL)')
        db.execute('INSERT INTO predictions(request_id,ticker,risk_level,payload) VALUES (?,?,?,?)',(prediction['request_id'],'SBER','low',json.dumps(prediction)))
    with TestClient(create_app(path)) as client:
        assert client.get('/predictions/'+prediction['request_id']).json()['risk_score'] == .185
        assert client.post('/predict',json=EXAMPLE).status_code == 200
