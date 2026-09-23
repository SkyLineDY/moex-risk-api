"""Capture actual HTTP outcomes from the running application, without third-party dependencies."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8001"
payload = json.loads((ROOT / "examples/low.json").read_text(encoding="utf-8"))
rows = []


def check(name, method, path, expected, data=None):
    body = json.dumps(data).encode() if data is not None else None
    request = Request(BASE + path, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as exc:
        response = exc
    with response:
        value = json.load(response)
        actual = response.status
    rows.append({"case": name, "method": method, "path": path, "expected": expected,
                 "actual": actual, "passed": actual == expected})
    assert actual == expected, (name, actual, value)
    return value


check("Проверка сервиса", "GET", "/health", 200)
check("Информация о модели", "GET", "/model-info", 200)
result = check("Корректная оценка", "POST", "/predict", 201, payload)
check("Отрицательная волатильность", "POST", "/predict", 422, payload | {"annual_volatility_pct": -10})
check("Неизвестный тикер", "POST", "/predict", 400, payload | {"ticker": "UNKNOWN"})
check("Существующий UUID", "GET", "/predictions/" + result["request_id"], 200)
check("Несуществующий UUID", "GET", "/predictions/00000000-0000-0000-0000-000000000000", 404)
check("Ограничение выдачи", "GET", "/predictions?limit=2", 200)
check("Фильтр высокого риска", "GET", "/predictions?risk_level=high", 200)
check("Некорректный limit", "GET", "/predictions?limit=-5", 422)
check("Неизвестный режим торгов", "POST", "/predict", 400, payload | {"board": "TEST"})
check("Неверная категория", "GET", "/predictions?risk_level=unknown", 422)
out = ROOT / "artifacts/test-results.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"base_url": BASE, "cases": rows, "sample_prediction": result}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(rows)} live HTTP checks passed")
