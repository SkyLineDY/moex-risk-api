import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_batch_size: int = 8
    batch_wait_ms: float = 50
    queue_size: int = 100
    inference_timeout: float = 2
    model_overhead: float = 0.020
    model_per_item: float = 0.003
    llm_base_url: str = "https://vireonix.ai/v1"
    llm_model: str = "auto"
    llm_timeout: float = 60
    llm_api_key: str = ""

    @classmethod
    def from_env(cls):
        return cls(llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
                   llm_model=os.getenv("LLM_MODEL", cls.llm_model),
                   llm_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
                   llm_api_key=os.getenv("LLM_API_KEY", ""))

    def __post_init__(self):
        if self.max_batch_size < 1 or self.queue_size < 1 or self.batch_wait_ms < 0:
            raise ValueError("Invalid queue settings")
        if min(self.inference_timeout, self.llm_timeout) <= 0:
            raise ValueError("Timeouts must be positive")
