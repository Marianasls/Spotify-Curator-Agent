import time
from typing import Any, Dict, List
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class ExecutionMetricsCallback(BaseCallbackHandler):
    """
    Callback Handler para agregaçao de tokens e contagem de chamadas ao LLM
    durante o ciclo de vida de uma única requisição do usuário.
    """
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.llm_latency_seconds = 0.0
        self._call_starts: dict[UUID, float] = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        # Marca o início exato da chamada de inferência ao Gemini
        self._call_starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        # Soma apenas o tempo que o Gemini levou para responder
        start = self._call_starts.pop(run_id, None)
        if start is not None:
            self.llm_latency_seconds += time.perf_counter() - start

        self.llm_calls += 1
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message and hasattr(message, "usage_metadata") and message.usage_metadata:
                    usage = message.usage_metadata
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)

