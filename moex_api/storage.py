"""One SQLite connection per operation; history survives application restart."""
import sqlite3
from pathlib import Path

from .schemas import PredictionResponse


class PredictionStorage:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                ticker TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                payload TEXT NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON predictions(ticker)")

    def ping(self):
        with self.connect() as connection:
            connection.execute("SELECT seq FROM predictions LIMIT 1").fetchone()

    def save(self, prediction: PredictionResponse):
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO predictions(request_id,ticker,risk_level,payload) VALUES (?,?,?,?)",
                (str(prediction.request_id), prediction.ticker, prediction.risk_level.value,
                 prediction.model_dump_json()),
            )

    def get(self, request_id: str):
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM predictions WHERE request_id=?", (request_id,)).fetchone()
        return PredictionResponse.model_validate_json(row[0]) if row else None

    def list(self, limit: int, offset: int, risk_level=None, ticker=None):
        clauses, values = [], []
        if risk_level is not None:
            clauses.append("risk_level=?")
            values.append(risk_level.value)
        if ticker is not None:
            clauses.append("ticker=?")
            values.append(ticker)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM predictions" + where + " ORDER BY seq DESC LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
        return [PredictionResponse.model_validate_json(row[0]) for row in rows]
