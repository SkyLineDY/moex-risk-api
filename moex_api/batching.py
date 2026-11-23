import asyncio
import logging
from dataclasses import dataclass
from time import monotonic

from fastapi import HTTPException

logger = logging.getLogger("moex_api")


@dataclass
class InferenceTask:
    data: object
    trace_id: str
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        self.queue = asyncio.Queue(maxsize=settings.queue_size)
        self.worker = None
        self.processed_batches = self.processed_items = self.last_batch_size = self.max_observed = 0

    def start(self):
        self.worker = asyncio.create_task(self.run(), name="moex-batch-worker")

    async def close(self):
        self.worker.cancel()
        try:
            await self.worker
        except asyncio.CancelledError:
            pass
        while not self.queue.empty():
            task = self.queue.get_nowait()
            if not task.future.done():
                task.future.set_exception(HTTPException(503, "Сервис останавливается"))
            self.queue.task_done()

    async def submit(self, data, trace_id):
        if self.worker is None or self.worker.done():
            raise HTTPException(503, "Обработчик очереди недоступен")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(InferenceTask(data, trace_id, future))
        except asyncio.QueueFull:
            raise HTTPException(429, "Очередь заполнена", headers={"Retry-After": "1"})
        logger.info("stage=queued trace=%s size=%s", trace_id, self.queue.qsize())
        try:
            return await asyncio.wait_for(future, self.settings.inference_timeout)
        except TimeoutError:
            raise HTTPException(504, "Истекло время ожидания расчёта")
        finally:
            if not future.done():
                future.cancel()

    async def run(self):
        while True:
            tasks = [await self.queue.get()]
            try:
                deadline = monotonic() + self.settings.batch_wait_ms / 1000
                while len(tasks) < self.settings.max_batch_size:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    try:
                        tasks.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except TimeoutError:
                        break
                active = [task for task in tasks if not task.future.done()]
                if active:
                    results = await self.model.predict_many([t.data for t in active], [t.trace_id for t in active], "queued")
                    if len(results) != len(active):
                        raise ValueError("Model returned an unexpected batch size")
                    self.last_batch_size = len(active)
                    self.max_observed = max(self.max_observed, len(active))
                    self.processed_batches += 1
                    self.processed_items += len(active)
                    for task, result in zip(active, results, strict=True):
                        if not task.future.done():
                            task.future.set_result(result)
            except asyncio.CancelledError:
                for task in tasks:
                    if not task.future.done():
                        task.future.set_exception(HTTPException(503, "Сервис останавливается"))
                raise
            except Exception:
                logger.exception("Batch calculation failed")
                for task in tasks:
                    if not task.future.done():
                        task.future.set_exception(HTTPException(503, "Ошибка пакетного расчёта"))
            finally:
                for _ in tasks:
                    self.queue.task_done()

    def status(self):
        return {"queue_size": self.queue.qsize(), "queue_max_size": self.settings.queue_size,
                "max_batch_size": self.settings.max_batch_size, "max_batch_wait_ms": self.settings.batch_wait_ms,
                "inference_timeout_seconds": self.settings.inference_timeout,
                "worker_running": self.worker is not None and not self.worker.done(),
                "processed_batches": self.processed_batches, "processed_items": self.processed_items,
                "last_batch_size": self.last_batch_size, "max_observed_batch_size": self.max_observed}
