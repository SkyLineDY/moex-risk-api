import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from . import services
from .model import MODEL_VERSION
from .schemas import (ErrorResponse, HealthResponse, Instrument, ModelInfo,
                      PredictionResponse, RiskLevel, StockRequest)
from .storage import PredictionStorage

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("moex_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_app(db_path=None, model_ready=None):
    storage = PredictionStorage(Path(db_path or os.getenv("MOEX_DB_PATH", ROOT / "data/predictions.sqlite3")))

    @asynccontextmanager
    async def lifespan(app):
        storage.initialize()
        yield

    app = FastAPI(
        title="MOEX Risk API", version="1.0.0", lifespan=lifespan,
        description="REST API оценки риска акций Мосбиржи. "
                    "Детерминированная модель, пользовательские показатели, сохранение в SQLite. "
                    "Балл не является вероятностью убытка или инвестиционной рекомендацией.",
    )
    app.state.model_ready = (os.getenv("MOEX_MODEL_READY", "1") == "1") if model_ready is None else model_ready
    app.state.storage = storage

    @app.middleware("http")
    async def observe(request: Request, call_next):
        trace_id, start = str(uuid4()), perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - start) * 1000:.2f}"
        logger.info("trace=%s method=%s path=%s status=%s duration_ms=%s", trace_id,
                    request.method, request.url.path, response.status_code, response.headers["X-Process-Time-Ms"])
        return response

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request, exc):
        logger.error("SQLite operation failed: %s", type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": "Хранилище временно недоступно"})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/health", response_model=HealthResponse, tags=["Сервис"],
             summary="Проверка работоспособности", description="Проверяет доступность модели и таблицы SQLite.",
             responses={503: {"model": ErrorResponse}})
    def health():
        storage.ping()
        if not app.state.model_ready:
            raise HTTPException(503, "Модель временно недоступна")
        return {"status": "ok", "model_ready": True, "storage": "sqlite"}

    @app.get("/model-info", response_model=ModelInfo, tags=["Сервис"],
             summary="Информация о модели", description="Тип, версия, признаки и пороги модели риска.")
    def model_info():
        return ModelInfo(name="moex-risk-baseline", version=MODEL_VERSION,
                         model_type="deterministic_heuristic", ready=app.state.model_ready,
                         description="Фиксированная взвешенная формула. Обучение на биржевой истории не проводилось.",
                         thresholds={"low": "[0, 0.35)", "medium": "[0.35, 0.65)", "high": "[0.65, 1]"},
                         features=["annual_volatility_pct", "max_drawdown_pct",
                                   "avg_daily_turnover_mrub", "bid_ask_spread_pct"])

    @app.get("/instruments", response_model=list[Instrument], tags=["Сервис"],
             summary="Справочник инструментов", description="Фиксированный список из пяти тикеров; не полный текущий листинг биржи.")
    def instruments():
        return [Instrument(ticker=t, name=n) for t, n in services.INSTRUMENTS.items()]

    @app.post("/predict", response_model=PredictionResponse, status_code=201, tags=["Прогнозы"],
              summary="Оценить риск акции", description="Вычисляет балл и сохраняет новый результат. Location указывает созданный ресурс.",
              responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def predict(data: StockRequest, response: Response):
        result = services.predict(data, storage, app.state.model_ready)
        response.headers["Location"] = f"/predictions/{result.request_id}"
        return result

    @app.get("/predictions", response_model=list[PredictionResponse], tags=["Прогнозы"],
             summary="История оценок", description="Новые записи первыми; фильтры применяются до limit и offset.",
             responses={503: {"model": ErrorResponse}})
    def predictions(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0),
                    risk_level: RiskLevel | None = None,
                    ticker: str | None = Query(None, min_length=1, max_length=12, pattern=r"^[A-Za-z0-9]+$")):
        return storage.list(limit, offset, risk_level, ticker.upper() if ticker else None)

    @app.get("/predictions/{request_id}", response_model=PredictionResponse, tags=["Прогнозы"],
             summary="Получить сохранённую оценку", description="Поиск по UUID, возвращённому POST /predict.",
             responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def prediction(request_id: UUID):
        result = storage.get(str(request_id))
        if result is None:
            raise HTTPException(404, "Прогноз не найден")
        return result

    return app
