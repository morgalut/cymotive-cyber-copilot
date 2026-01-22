# cymotive-cyber-copilot/app/services/copilot_service.py

import time
from typing import Any, Dict

from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService


class CopilotService:
    def __init__(self, retrieval: RetrievalService, llm: LLMService):
        self.retrieval = retrieval
        self.llm = llm

    def analyze(
        self,
        report_text: str,
        top_k: int,
        use_hybrid: bool,
        response_style: str,
    ) -> Dict[str, Any]:

        print("\n[COPILOT] ===== analyze() START =====")
        print(f"[COPILOT] report_text_len={len(report_text)}")
        print(f"[COPILOT] top_k={top_k} use_hybrid={use_hybrid} response_style={response_style}")

        t0 = time.time()

  
        # Retrieval step (RAG)
  
        print("[COPILOT] → Calling RetrievalService.search()")

        retrieved, rmeta = self.retrieval.search(
            report_text,
            top_k=top_k,
            use_hybrid=use_hybrid,
        )

        print(f"[COPILOT] ← Retrieval returned {len(retrieved)} incidents")
        for i, inc in enumerate(retrieved, 1):
            print(
                f"  [COPILOT][RETRIEVAL] #{i} "
                f"id={inc.get('id')} score={inc.get('score'):.3f} "
                f"title={inc.get('title')}"
            )

  
        # LLM step
  
        print("[COPILOT] → Calling LLMService.analyze()")
        print(f"[COPILOT] LLM provider={self.llm.provider} model={getattr(self.llm, 'model', None)}")

        result = self.llm.analyze(
            report_text=report_text,
            retrieved_incidents=retrieved,
            response_style=response_style,
        )

        print("[COPILOT] ← LLM returned response")
        print(f"[COPILOT] LLM output keys={list(result.keys())}")

  
        # Metadata enrichment
  
        latency_ms = int((time.time() - t0) * 1000)

        print(f"[COPILOT] Total analyze latency={latency_ms}ms")

        result.setdefault("metadata", {})
        result["metadata"].update({
            "latency_ms": latency_ms,
            "retrieval": rmeta,
            "llm_provider": self.llm.provider,
            "model": getattr(self.llm, "model", None),
        })

        print("[COPILOT] Metadata attached")
        print("[COPILOT] ===== analyze() END =====\n")

        return result
