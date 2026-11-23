import logging
import math
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import services
from .batching import DynamicBatcher
from .config import Settings
from .llm import LLMClient
from .model import MODEL_VERSION
from .schemas import (BatchRequest, BatchResponse, ErrorResponse, ExplainedResponse,
                      HealthResponse, Instrument, LLMRequest, LLMResponse, ModelInfo,
                      PaginatedResponse, PredictionResponse, QueueStatus, RiskLevel, StockRequest)
from .storage import PredictionStorage, WEEKDAYS

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("moex_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_app(db_path=None, model_ready=None, settings=None, llm_transport=None):
    settings = settings or Settings.from_env()
    storage = PredictionStorage(db_path or os.getenv("MOEX_DB_PATH", ROOT / "data/predictions.sqlite3"))

    @asynccontextmanager
    async def lifespan(app):
        await storage.initialize()
        app.state.llm = LLMClient(settings, llm_transport)
        app.state.batcher = DynamicBatcher(services.InferenceModel(settings), settings)
        app.state.batcher.start()
        try:
            yield
        finally:
            await app.state.batcher.close()
            await app.state.llm.close()

    app = FastAPI(title="MOEX Risk API", version="2.0.0", lifespan=lifespan,
                  description="Оценка риска акций: одиночный и пакетный расчёт, очередь, история SQLite и текстовые объяснения.")
    app.state.model_ready = os.getenv("MOEX_MODEL_READY", "1") == "1" if model_ready is None else model_ready
    app.state.storage = storage
    app.state.settings = settings

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = perf_counter()
        candidate = request.headers.get("X-Request-ID", "")
        valid = re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", candidate)
        trace_id = candidate if valid else str(uuid4())
        request.state.request_id = trace_id
        logger.info("stage=start trace=%s method=%s path=%s", trace_id, request.method, request.url.path)
        try:
            response = await call_next(request) if not candidate or valid else JSONResponse(
                status_code=400, content={"detail": "Некорректный X-Request-ID", "request_id": trace_id})
        except Exception:
            logger.exception("stage=error trace=%s", trace_id)
            response = JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера", "request_id": trace_id})
        response.headers["X-Request-ID"] = trace_id
        elapsed = perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.6f}"
        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"
        logger.info("stage=complete trace=%s status=%s duration_ms=%s", trace_id, response.status_code, response.headers["X-Process-Time-Ms"])
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": exc.detail, "request_id": request.state.request_id}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = [{k: v for k, v in item.items() if k not in {"input", "ctx"}} for item in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors), "request_id": request.state.request_id})

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request, exc):
        logger.error("stage=storage_error trace=%s type=%s", request.state.request_id, type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": "Хранилище временно недоступно", "request_id": request.state.request_id})

    def check_model():
        if not app.state.model_ready:
            raise HTTPException(503, "Модель временно недоступна")

    async def calculate(data, trace, source="single"):
        check_model()
        services.validate(data)
        return (await app.state.batcher.model.predict_many([data], [trace], source))[0]

    async def save(result):
        await storage.save(result)
        logger.info("stage=saved trace=%s prediction=%s", result.trace_id, result.prediction_id)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/health", response_model=HealthResponse, tags=["Сервис"], summary="Состояние модели и SQLite", responses={503: {"model": ErrorResponse}})
    async def health():
        await storage.ping()
        check_model()
        if not app.state.batcher.status()["worker_running"]:
            raise HTTPException(503, "Обработчик очереди недоступен")
        return {"status": "ok", "model_ready": True, "storage": "sqlite"}

    @app.get("/model-info", response_model=ModelInfo, tags=["Сервис"], summary="Версия и признаки модели")
    async def model_info():
        return ModelInfo(name="moex-risk-baseline", version=MODEL_VERSION, model_type="deterministic_heuristic",
                         ready=app.state.model_ready, description="Взвешенная формула с фиксированными коэффициентами.",
                         thresholds={"low": "[0, 0.35)", "medium": "[0.35, 0.65)", "high": "[0.65, 1]"},
                         features=["annual_volatility_pct", "max_drawdown_pct", "avg_daily_turnover_mrub", "bid_ask_spread_pct"])

    @app.get("/instruments", response_model=list[Instrument], tags=["Сервис"], summary="Поддерживаемые инструменты")
    async def instruments():
        return [Instrument(ticker=t, name=n) for t,n in services.INSTRUMENTS.items()]

    @app.post("/predict", response_model=PredictionResponse, status_code=200, tags=["Расчёт"], summary="Рассчитать риск акции", responses={400:{"model":ErrorResponse},503:{"model":ErrorResponse}})
    async def predict(data: StockRequest, request: Request, response: Response):
        result = await calculate(data, request.state.request_id)
        await save(result)
        response.headers["Location"] = f"/predictions/{result.prediction_id}"
        return result

    @app.post("/batch-predict", response_model=BatchResponse, status_code=200, tags=["Расчёт"], summary="Рассчитать пакет акций", description="До восьми объектов. Все результаты сохраняются одной транзакцией.", responses={400:{"model":ErrorResponse},503:{"model":ErrorResponse}})
    async def batch_predict(data: BatchRequest, request: Request):
        start = perf_counter()
        check_model()
        for item in data.items:
            services.validate(item)
        results = await app.state.batcher.model.predict_many(data.items, [request.state.request_id]*len(data.items), "batch")
        await storage.save_many(results)
        logger.info("stage=saved_batch trace=%s count=%s", request.state.request_id, len(results))
        return BatchResponse(request_id=request.state.request_id, batch_size=len(results),
                             processing_time_ms=round((perf_counter()-start)*1000,2), items=results)

    @app.post("/predict-queued", response_model=PredictionResponse, status_code=200, tags=["Расчёт"], summary="Рассчитать через очередь", responses={400:{"model":ErrorResponse},429:{"model":ErrorResponse},503:{"model":ErrorResponse},504:{"model":ErrorResponse}})
    async def predict_queued(data: StockRequest, request: Request, response: Response):
        check_model()
        services.validate(data)
        result = await app.state.batcher.submit(data, request.state.request_id)
        await save(result)
        response.headers["Location"] = f"/predictions/{result.prediction_id}"
        return result

    @app.get("/queue-status", response_model=QueueStatus, tags=["Сервис"], summary="Очередь и статистика пакетов")
    async def queue_status():
        return app.state.batcher.status()

    @app.get("/predictions", response_model=list[PredictionResponse], tags=["История"], summary="История с фильтрами", responses={503:{"model":ErrorResponse}})
    async def predictions(limit: int=Query(10, ge=1, le=100), offset: int=Query(0,ge=0), risk_level: RiskLevel|None=None,
                          ticker: str|None=Query(None,min_length=1,max_length=12,pattern=r"^[A-Za-z0-9]+$")):
        return await storage.list(limit,offset,risk_level,ticker.upper() if ticker else None)

    @app.get("/predictions/by-weekday", response_model=PaginatedResponse, tags=["История"], summary="Страницы истории по дню недели", description="День недели определяется по времени создания в UTC.", responses={400:{"model":ErrorResponse},404:{"model":ErrorResponse},503:{"model":ErrorResponse}})
    async def weekday_page(request: Request, weekday: str, page: int=Query(1,ge=1), per_page: int=Query(5,ge=1,le=100)):
        weekday = weekday.lower()
        if weekday not in WEEKDAYS:
            raise HTTPException(400, "Неизвестный день недели")
        data,total = await storage.weekday_page(weekday,page,per_page)
        pages = math.ceil(total/per_page)
        if page > max(pages,1):
            raise HTTPException(404, "Страница не найдена")
        url = request.url.include_query_params(weekday=weekday,per_page=per_page)
        return {"data":data,"meta":{"weekday":weekday,"current_page":page,"per_page":per_page,
                "total_pages":pages,"total_items":total,"timezone":"UTC",
                "next_page":str(url.include_query_params(page=page+1)) if page<pages else None,
                "prev_page":str(url.include_query_params(page=page-1)) if page>1 else None}}

    @app.get("/requests/{request_id}/predictions", response_model=list[PredictionResponse], tags=["История"], summary="Результаты HTTP запроса", responses={404:{"model":ErrorResponse},503:{"model":ErrorResponse}})
    async def trace_predictions(request_id: str):
        results = await storage.by_trace(request_id)
        if not results:
            raise HTTPException(404, "Результаты запроса не найдены")
        return results

    @app.get("/predictions/{request_id}", response_model=PredictionResponse, tags=["История"], summary="Результат по идентификатору", responses={409:{"model":ErrorResponse},404:{"model":ErrorResponse},503:{"model":ErrorResponse}})
    async def prediction(request_id: str):
        matches = await storage.by_trace(request_id)
        if len(matches) > 1:
            raise HTTPException(409, "Запрос содержит несколько оценок")
        if matches:
            return matches[0]
        try:
            prediction_id = UUID(request_id)
        except ValueError:
            prediction_id = None
        result = await storage.get(str(prediction_id)) if prediction_id else None
        if result is None:
            raise HTTPException(404, "Прогноз не найден")
        return result

    @app.post("/llm-chat", response_model=LLMResponse, include_in_schema=False)
    @app.post("/llm/chat", response_model=LLMResponse, tags=["LLM"], summary="Запрос к языковой модели", responses={502:{"model":ErrorResponse},504:{"model":ErrorResponse}})
    async def llm_chat(data: LLMRequest, request: Request):
        return await app.state.llm.chat(data, request.state.request_id)

    @app.post("/predict-with-llm", response_model=ExplainedResponse, status_code=200, tags=["LLM"], summary="Расчёт с текстовым объяснением", description="При ошибке LLM оценка не сохраняется.", responses={400:{"model":ErrorResponse},502:{"model":ErrorResponse},503:{"model":ErrorResponse},504:{"model":ErrorResponse}})
    async def explain(data: StockRequest, request: Request, response: Response):
        result = await calculate(data, request.state.request_id, "llm")
        prompt = ("На русском кратко опиши вклад факторов в балл риска. Только факты и числа из JSON. "
                  "Не добавляй вступление, вывод, советы, предостережения и рекомендации о покупке. "
                  "Балл уже рассчитан; не меняй его. JSON: " + result.model_dump_json())
        explanation = await app.state.llm.chat(LLMRequest(prompt=prompt), request.state.request_id)
        await save(result)
        response.headers["Location"] = f"/predictions/{result.prediction_id}"
        return ExplainedResponse(prediction=result,llm=explanation)

    return app
