from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


EXAMPLE = {
    "ticker": "SBER", "board": "TQBR", "annual_volatility_pct": 24,
    "max_drawdown_pct": 12, "avg_daily_turnover_mrub": 150,
    "bid_ask_spread_pct": 0.1, "observation_days": 60,
}


class StockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False,
                              json_schema_extra={"examples": [EXAMPLE]})
    ticker: str = Field(min_length=1, max_length=12, pattern=r"^[A-Z0-9]+$",
                        description="Тикер из справочника /instruments")
    board: str = Field(default="TQBR", min_length=1, max_length=8,
                       description="Режим торгов; прототип поддерживает TQBR")
    annual_volatility_pct: float = Field(ge=0, le=300, description="Годовая волатильность, %")
    max_drawdown_pct: float = Field(ge=0, le=100, description="Максимальная просадка за период, %")
    avg_daily_turnover_mrub: float = Field(ge=0, le=1_000_000, description="Средний дневной оборот, млн руб.")
    bid_ask_spread_pct: float = Field(ge=0, le=100, description="Спред между лучшими ценами покупки и продажи, %")
    observation_days: int = Field(ge=20, le=252, strict=True,
                                  description="Число торговых дней в выборке, от 20 до 252")

    @field_validator("ticker", "board", mode="before")
    @classmethod
    def normalize_code(cls, value):
        return value.strip().upper() if isinstance(value, str) else value


class Factor(BaseModel):
    name: str
    normalized_value: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    contribution: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    request_id: UUID
    created_at: datetime
    ticker: str
    board: str
    risk_score: float = Field(ge=0, le=1, description="Балл риска, не вероятность убытка")
    risk_level: RiskLevel
    recommendation: str
    model_version: str
    input_data: StockRequest
    factors: list[Factor]
    data_source: str = "user_provided"


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    storage: str


class Instrument(BaseModel):
    ticker: str
    name: str
    board: str = "TQBR"


class ModelInfo(BaseModel):
    name: str
    version: str
    model_type: str
    ready: bool
    description: str
    thresholds: dict[str, str]
    features: list[str]
