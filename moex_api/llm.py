import logging
from time import perf_counter

import httpx
from fastapi import HTTPException

from .schemas import LLMResponse

logger = logging.getLogger("moex_api")


class LLMClient:
    def __init__(self, settings, transport=None):
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        self.client = httpx.AsyncClient(base_url=settings.llm_base_url.rstrip('/') + '/',
                                        timeout=settings.llm_timeout, headers=headers, transport=transport)

    async def close(self):
        await self.client.aclose()

    async def chat(self, data, trace_id):
        start = perf_counter()
        model = data.model or self.settings.llm_model
        try:
            response = await self.client.post("chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": data.prompt}],
                "temperature": data.temperature, "max_tokens": data.max_tokens,
            }, headers={"X-Request-ID": trace_id})
            response.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(504, "Истекло время ожидания LLM")
        except httpx.HTTPError:
            logger.warning("stage=llm trace=%s status=upstream_error", trace_id)
            raise HTTPException(502, "Сервис LLM недоступен")
        try:
            body = response.json()
            choice = body["choices"][0]
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty content")
            result = LLMResponse(request_id=trace_id, requested_model=model,
                                 returned_model=body.get("model"), answer=answer,
                                 finish_reason=choice.get("finish_reason"), usage=body.get("usage"),
                                 latency_ms=round((perf_counter() - start) * 1000, 2))
        except (ValueError, KeyError, IndexError, TypeError):
            raise HTTPException(502, "Некорректный ответ сервиса LLM")
        logger.info("stage=llm trace=%s status=200 duration_ms=%s", trace_id, result.latency_ms)
        return result
