from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from .model import MODEL_VERSION, calculate_risk, classify
from .schemas import PredictionResponse, StockRequest

INSTRUMENTS = {
    "SBER": "Сбербанк", "GAZP": "Газпром", "LKOH": "ЛУКОЙЛ",
    "MOEX": "Московская биржа", "GMKN": "Норильский никель",
}


def predict(data: StockRequest, storage, model_ready: bool):
    if not model_ready:
        raise HTTPException(503, "Модель временно недоступна")
    if data.ticker not in INSTRUMENTS:
        raise HTTPException(400, "Тикер отсутствует в справочнике /instruments")
    if data.board != "TQBR":
        raise HTTPException(400, "Поддерживается только режим TQBR")
    score, factors = calculate_risk(data)
    level = classify(score)
    notes = {
        "low": "Низкий балл риска; проверьте качество исходных показателей.",
        "medium": "Средний балл риска; рассмотрите вклад отдельных факторов.",
        "high": "Высокий балл риска; требуется дополнительный анализ факторов.",
    }
    result = PredictionResponse(
        request_id=uuid4(), created_at=datetime.now(timezone.utc), ticker=data.ticker,
        board=data.board, risk_score=score, risk_level=level, model_version=MODEL_VERSION,
        recommendation=notes[level.value], input_data=data, factors=factors,
    )
    storage.save(result)
    return result
