"""Deterministic scoring baseline; no training or market feed."""
from .schemas import Factor, RiskLevel, StockRequest

MODEL_VERSION = "1.0.0"


def classify(score: float) -> RiskLevel:
    if score < 0.35:
        return RiskLevel.low
    if score < 0.65:
        return RiskLevel.medium
    return RiskLevel.high


def calculate_risk(data: StockRequest) -> tuple[float, list[Factor]]:
    raw = [
        ("volatility", min(data.annual_volatility_pct / 80, 1), 0.4),
        ("drawdown", min(data.max_drawdown_pct / 60, 1), 0.3),
        ("illiquidity", 1 - min(data.avg_daily_turnover_mrub / 100, 1), 0.2),
        ("spread", min(data.bid_ask_spread_pct / 2, 1), 0.1),
    ]
    factors = [Factor(name=name, normalized_value=round(value, 6), weight=weight,
                      contribution=round(value * weight, 6)) for name, value, weight in raw]
    return round(sum(value * weight for _, value, weight in raw), 6), factors
