import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from .model import MODEL_VERSION, calculate_risk, classify
from .schemas import PredictionResponse

logger = logging.getLogger("moex_api")
INSTRUMENTS = {"SBER": "Сбербанк", "GAZP": "Газпром", "LKOH": "ЛУКОЙЛ",
               "MOEX": "Московская биржа", "GMKN": "Норильский никель"}


def validate(data):
    if data.ticker not in INSTRUMENTS:
        raise HTTPException(400, "Тикер отсутствует в справочнике /instruments")
    if data.board != "TQBR":
        raise HTTPException(400, "Поддерживается только режим TQBR")


def make_prediction(data, trace_id, source):
    score, factors = calculate_risk(data)
    level = classify(score)
    notes = {"low": "Низкий балл риска.", "medium": "Средний балл риска.", "high": "Высокий балл риска."}
    return PredictionResponse(request_id=trace_id, prediction_id=uuid4(), created_at=datetime.now(timezone.utc),
                              ticker=data.ticker, board=data.board, risk_score=score,
                              risk_level=level, recommendation=notes[level.value],
                              model_version=MODEL_VERSION, input_data=data, factors=factors,
                              trace_id=trace_id, source=source)


class InferenceModel:
    def __init__(self, settings):
        self.settings = settings

    async def predict_many(self, items, trace_ids, source):
        # This delay models invocation overhead; arithmetic remains synchronous.
        await asyncio.sleep(self.settings.model_overhead + self.settings.model_per_item * len(items))
        results = [make_prediction(data, trace, source) for data, trace in zip(items, trace_ids, strict=True)]
        for result in results:
            logger.info("stage=inference trace=%s prediction=%s source=%s", result.trace_id, result.prediction_id, source)
        return results
