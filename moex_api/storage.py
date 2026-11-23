from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from .schemas import PredictionResponse

WEEKDAYS = dict(zip(("sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"), range(7)))


class PredictionStorage:
    def __init__(self, path):
        self.path = Path(path)

    @asynccontextmanager
    async def connect(self):
        async with aiosqlite.connect(self.path, timeout=5) as db:
            await db.execute("PRAGMA busy_timeout=5000")
            yield db

    async def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("CREATE TABLE IF NOT EXISTS predictions (seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE, ticker TEXT NOT NULL, risk_level TEXT NOT NULL, payload TEXT NOT NULL)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON predictions(ticker)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trace ON predictions(json_extract(payload, '$.trace_id'))")
            await db.commit()

    async def ping(self):
        async with self.connect() as db:
            async with db.execute("SELECT seq FROM predictions LIMIT 1") as cursor:
                await cursor.fetchone()

    async def save(self, prediction):
        await self.save_many([prediction])

    async def save_many(self, predictions):
        async with self.connect() as db:
            await db.executemany("INSERT INTO predictions(request_id,ticker,risk_level,payload) VALUES (?,?,?,?)", [(str(p.prediction_id), p.ticker, p.risk_level.value, p.model_dump_json()) for p in predictions])
            await db.commit()

    async def get(self, request_id):
        async with self.connect() as db:
            async with db.execute("SELECT payload FROM predictions WHERE request_id=?", (request_id,)) as cursor:
                row = await cursor.fetchone()
        return PredictionResponse.model_validate_json(row[0]) if row else None

    async def by_trace(self, trace_id):
        async with self.connect() as db:
            async with db.execute("SELECT payload FROM predictions WHERE json_extract(payload,'$.trace_id')=? ORDER BY seq", (trace_id,)) as cursor:
                rows = await cursor.fetchall()
        return [PredictionResponse.model_validate_json(row[0]) for row in rows]

    async def list(self, limit, offset, risk_level=None, ticker=None):
        clauses, values = [], []
        if risk_level is not None:
            clauses.append("risk_level=?")
            values.append(risk_level.value)
        if ticker is not None:
            clauses.append("ticker=?")
            values.append(ticker)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        async with self.connect() as db:
            async with db.execute("SELECT payload FROM predictions" + where + " ORDER BY seq DESC LIMIT ? OFFSET ?", (*values, limit, offset)) as cursor:
                rows = await cursor.fetchall()
        return [PredictionResponse.model_validate_json(row[0]) for row in rows]

    async def weekday_page(self, weekday, page, per_page):
        where = " WHERE strftime('%w', json_extract(payload, '$.created_at'))=?"
        async with self.connect() as db:
            await db.execute("BEGIN")
            async with db.execute("SELECT COUNT(*) FROM predictions" + where, (str(WEEKDAYS[weekday]),)) as cursor:
                total = (await cursor.fetchone())[0]
            async with db.execute("SELECT payload FROM predictions" + where + " ORDER BY seq DESC LIMIT ? OFFSET ?", (str(WEEKDAYS[weekday]), per_page, (page - 1) * per_page)) as cursor:
                rows = await cursor.fetchall()
        return [PredictionResponse.model_validate_json(row[0]) for row in rows], total
