from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    request_id: str
    prediction_id: UUID
    created_at: datetime
    ticker: str
    board: str
    risk_score: float = Field(ge=0, le=1, description="Балл по взвешенной формуле")
    risk_level: RiskLevel
    recommendation: str
    model_version: str
    input_data: StockRequest
    factors: list[Factor]
    data_source: str = "user_provided"
    trace_id: str | None = None
    source: str = "single"

    @model_validator(mode="before")
    @classmethod
    def read_previous_record(cls, value):
        if isinstance(value, dict) and "prediction_id" not in value and "request_id" in value:
            return {**value, "prediction_id": value["request_id"],
                    "request_id": value.get("trace_id") or value["request_id"]}
        return value


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[StockRequest] = Field(min_length=1, max_length=8)


class BatchResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[PredictionResponse]


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=5000)
    model: str | None = Field(default=None, min_length=1, max_length=128)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=300, ge=1, le=2000)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Prompt is blank")
        return value


class LLMResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None = None
    answer: str
    finish_reason: str | None = None
    usage: dict | None = None
    latency_ms: float


class ExplainedResponse(BaseModel):
    prediction: PredictionResponse
    llm: LLMResponse


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    timezone: str = "UTC"
    next_page: str | None
    prev_page: str | None


class PaginatedResponse(BaseModel):
    data: list[PredictionResponse]
    meta: PaginationMeta


class QueueStatus(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: float
    inference_timeout_seconds: float
    worker_running: bool
    processed_batches: int
    processed_items: int
    last_batch_size: int
    max_observed_batch_size: int


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
