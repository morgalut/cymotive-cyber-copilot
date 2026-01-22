from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    IncidentRetrieveRequest,
    IncidentRetrieveResponse,
    IncidentAnalyzeRequest,
    IncidentAnalyzeResponse,
)
from app.services.copilot_service import CopilotService
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Singletons for PoC ---
retrieval_service = RetrievalService(settings.INCIDENTS_PATH)
retrieval_service.load_kb()
retrieval_service.build_index()

llm_service = LLMService(
    provider=settings.LLM_PROVIDER,
    openai_api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL,
)

copilot = CopilotService(retrieval=retrieval_service, llm=llm_service)


# ----------------------------
# Helpers: normalization
# ----------------------------
def _normalize_analyze_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj if isinstance(obj, dict) else {}

    obj.setdefault("summary", "")
    obj.setdefault("key_indicators", [])
    obj.setdefault("suspected_category", None)
    obj.setdefault("similar_incidents", [])
    obj.setdefault("assumptions", [])
    obj.setdefault("open_questions", [])
    obj.setdefault("metadata", {})

    mp = obj.setdefault("mitigation_plan", {})
    mp.setdefault("immediate", [])
    mp.setdefault("short_term", [])
    mp.setdefault("long_term", [])
    mp.setdefault("notes", [])

    return obj


# ----------------------------
# Routes
# ----------------------------
@router.post("/incidents/retrieve", response_model=IncidentRetrieveResponse)
def retrieve(req: IncidentRetrieveRequest, request: Request):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    try:
        matches, meta = retrieval_service.search(
            req.query,
            top_k=req.top_k,
            use_hybrid=req.use_hybrid,
        )
        meta.update(
            {
                "request_id": request_id,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        )
        return {"matches": matches, "metadata": meta}

    except Exception:
        logger.exception("Failed /incidents/retrieve request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Internal server error", "request_id": request_id},
        )


@router.post("/incidents/analyze", response_model=IncidentAnalyzeResponse)
def analyze(req: IncidentAnalyzeRequest, request: Request):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    print(f"[ROUTE] /incidents/analyze START request_id={request_id}")
    print(f"[ROUTE] Input report_text_len={len(req.report_text)} top_k={req.top_k} use_hybrid={req.use_hybrid}")

    try:
        raw = copilot.analyze(
            report_text=req.report_text,
            top_k=req.top_k,
            use_hybrid=req.use_hybrid,
            response_style=req.response_style,
        )

        print(f"[ROUTE] Copilot returned keys={list(raw.keys())}")

        out = _normalize_analyze_output(raw)

        latency_ms = int((time.time() - t0) * 1000)
        out["metadata"].update(
            {
                "request_id": request_id,
                "request_latency_ms": latency_ms,
            }
        )


        print(f"[ROUTE] /incidents/analyze END request_id={request_id} latency_ms={latency_ms}")
        return out

    except Exception as e:
        print(f"[ROUTE][ERROR] request_id={request_id} error={repr(e)}")
        raise
