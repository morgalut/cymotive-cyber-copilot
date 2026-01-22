# app/services/llm_service.py
import json
import time
from typing import Any, Dict, List, Optional
from app.core.config import settings

from app.services.prompts import SYSTEM_PROMPT, COMBINED_ANALYZE_PROMPT

try:
    from openai import OpenAI  
except Exception:
    OpenAI = None



def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default
    
def estimate_cost_usd(input_tokens: int, output_tokens: int, in_per_1k: float, out_per_1k: float) -> float:
    return round((input_tokens / 1000.0) * in_per_1k + (output_tokens / 1000.0) * out_per_1k, 6)


def normalize_usage(usage_dump: Optional[Dict[str, Any]], latency_ms: int) -> Dict[str, Any]:
    """
    Convert provider-specific usage into a stable summary + derived metrics.
    """
    usage_dump = usage_dump or {}

    input_tokens = _safe_int(usage_dump.get("input_tokens"), 0)
    output_tokens = _safe_int(usage_dump.get("output_tokens"), 0)
    total_tokens = _safe_int(usage_dump.get("total_tokens"), input_tokens + output_tokens)

    cached_input = _safe_int(
        (usage_dump.get("input_tokens_details") or {}).get("cached_tokens"), 0
    )
    reasoning_tokens = _safe_int(
        (usage_dump.get("output_tokens_details") or {}).get("reasoning_tokens"), 0
    )

    billable_input = max(input_tokens - cached_input, 0)

    latency_s = max(latency_ms / 1000.0, 1e-9)
    tokens_per_second = total_tokens / latency_s if total_tokens else 0.0

    ratio = (output_tokens / input_tokens) if input_tokens else None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input,
        "billable_input_tokens": billable_input,
        "reasoning_tokens": reasoning_tokens,
        "output_to_input_ratio": ratio,
        "tokens_per_second": round(tokens_per_second, 2),
    }



def incident_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "key_indicators",
            "suspected_category",
            "mitigation_plan",
            "similar_incidents",
            "assumptions",
            "open_questions",
        ],
        "properties": {
            "summary": {"type": "string"},
            "key_indicators": {"type": "array", "items": {"type": "string"}},
            "suspected_category": {
                "type": "string",
                "enum": [
                    "ransomware",
                    "phishing",
                    "malware",
                    "ddos",
                    "account_compromise",
                    "dns_tunneling",
                    "unknown",
                ],
            },
            "mitigation_plan": {
                "type": "object",
                "additionalProperties": False,
                "required": ["immediate", "short_term", "long_term", "notes"],
                "properties": {
                    "immediate": {"type": "array", "items": {"type": "string"}},
                    "short_term": {"type": "array", "items": {"type": "string"}},
                    "long_term": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
            },
            "similar_incidents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "title", "snippet", "score", "match_factors"],
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "snippet": {"type": "string"},
                        "score": {"type": "number"},
                        "match_factors": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
    }


class LLMService:
    def __init__(
        self,
        provider: str = "mock",
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.model = model
        self._client = None

        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not installed. Run: pip install openai")
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is missing")
            self._client = OpenAI(api_key=self.openai_api_key)

    def analyze(self, report_text: str, retrieved_incidents: List[Dict[str, Any]], response_style: str):
        print("[LLM] analyze() START")
        print(f"[LLM] provider={self.provider} model={self.model}")

        if self.provider == "mock":
            print("[LLM] Using MOCK response")
            return self._mock_response(report_text, retrieved_incidents)

        print("[LLM] Building prompt...")
        prompt = COMBINED_ANALYZE_PROMPT.format(
            report_text=report_text,
            retrieved_incidents_json=json.dumps(retrieved_incidents, ensure_ascii=False),
            response_style=response_style,
        )
        print(f"[LLM] Prompt length={len(prompt)} chars")

        print("[LLM] Calling OpenAI Responses API...")
        return self._openai_structured(prompt, retrieved_incidents, debug=True)



    def _openai_structured(
        self,
        user_prompt: str,
        retrieved: List[Dict[str, Any]],
        *,
        debug: bool = True,
    ) -> Dict[str, Any]:
        """
        Uses Responses API + Structured Outputs JSON schema.
        Adds step-by-step debug prints + robust parsing/grounding + token + cost telemetry.
        """

        def log(msg: str) -> None:
            if debug:
                print(msg)

        log("\n[LLM] ===== _openai_structured() START =====")
        log(f"[LLM] Provider=openai Model={self.model}")
        log(f"[LLM] Retrieved incidents count={len(retrieved)}")
        log(f"[LLM] Prompt length={len(user_prompt)} characters")
        log("[LLM] Structured Outputs: json_schema strict=true")

        t0 = time.time()

        # ---------- Step 1: OpenAI call ----------
        try:
            log("[LLM] → Sending request to OpenAI Responses API")
            resp = self._client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "incident_analysis",
                        "schema": incident_response_schema(),
                        "strict": True,
                    }
                },
            )
            latency_ms = int((time.time() - t0) * 1000)
            log(f"[LLM] ← OpenAI response received llm_latency_ms={latency_ms}")

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            log("[LLM][ERROR] OpenAI request failed")
            log(f"[LLM][ERROR] llm_latency_ms={latency_ms}")
            log(f"[LLM][ERROR] Exception type={type(e).__name__}")
            log(f"[LLM][ERROR] Message={e}")
            raise

        # ---------- Step 2: Extract raw output ----------
        raw_text = getattr(resp, "output_text", None) or ""
        log(f"[LLM] Raw output length={len(raw_text)}")
        if raw_text:
            log("[LLM] Raw output preview (first 300 chars):")
            log(raw_text[:300])
        else:
            log("[LLM][WARN] Empty output_text received from model")

        # ---------- Step 3: Parse JSON ----------
        try:
            data = json.loads(raw_text)
            log("[LLM] JSON parsed successfully")
            if isinstance(data, dict):
                log(f"[LLM] Parsed keys={list(data.keys())}")
            else:
                log(f"[LLM][WARN] Parsed JSON is not an object, type={type(data).__name__}")
        except Exception as e:
            log("[LLM][ERROR] Failed to parse JSON from model output")
            log(f"[LLM][ERROR] Exception type={type(e).__name__}")
            log(f"[LLM][ERROR] Message={e}")
            raise ValueError("Model output was not valid JSON") from e

        # ---------- Step 4: Defensive normalization ----------
        if not isinstance(data, dict):
            data = {}

        data.setdefault("summary", "")
        data.setdefault("key_indicators", [])
        data.setdefault("suspected_category", "unknown")
        data.setdefault("assumptions", [])
        data.setdefault("open_questions", [])

        mp = data.get("mitigation_plan") if isinstance(data.get("mitigation_plan"), dict) else {}
        mp.setdefault("immediate", [])
        mp.setdefault("short_term", [])
        mp.setdefault("long_term", [])
        mp.setdefault("notes", [])
        data["mitigation_plan"] = mp

        # ---------- Step 5: Grounding enforcement ----------
        log("[LLM] Enforcing grounding: overriding similar_incidents with retrieved results")
        data["similar_incidents"] = retrieved

        # ---------- Step 6: Usage normalization ----------
        usage = getattr(resp, "usage", None)
        usage_dump = None
        if usage is not None:
            usage_dump = usage.model_dump() if hasattr(usage, "model_dump") else usage
            log("[LLM] Token usage available")
        else:
            log("[LLM] Token usage not available")

        usage_norm = normalize_usage(usage_dump, latency_ms)

        # ---------- Step 7: Cost estimate ----------
        # Uses billable input tokens (input minus cached) + output tokens.
        cost_estimate = estimate_cost_usd(
            usage_norm.get("billable_input_tokens", 0),
            usage_norm.get("output_tokens", 0),
            settings.COST_PER_1K_INPUT_TOKENS_USD,
            settings.COST_PER_1K_OUTPUT_TOKENS_USD,
        )
        log(f"[LLM] Cost estimate (USD) = {cost_estimate}")

        # ---------- Step 8: Attach metadata (best presentation) ----------
        data.setdefault("metadata", {})
        data["metadata"].update(
            {
                "llm_latency_ms": latency_ms,
                "llm_provider": "openai",
                "model": self.model,

                # Raw provider usage for debugging
                "usage_raw": usage_dump,

                # Normalized stable usage for monitoring
                "usage": usage_norm,

                # Cost estimate (configurable via settings/.env)
                "cost_estimate_usd": cost_estimate,

                "structured_outputs": True,
            }
        )

        log("[LLM] Metadata attached")
        log("[LLM] ===== _openai_structured() END =====\n")

        return data


    

    def _mock_response(self, report_text: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Keep mock, but make it *slightly* smarter if you want:
        # For now, it's deterministic but at least returns RAG results.
        return {
            "summary": "Preliminary incident summary based on provided report. Further details required.",
            "key_indicators": [],
            "suspected_category": "unknown",
            "mitigation_plan": {
                "immediate": [
                    "Preserve logs and isolate affected hosts if confirmed compromised.",
                    "Block suspicious indicators if present (domains/IPs/hashes).",
                ],
                "short_term": [
                    "Validate scope and impact; identify entry point and affected accounts.",
                    "Run endpoint scans and confirm persistence mechanisms.",
                ],
                "long_term": [
                    "Harden access controls (MFA, least privilege) and improve detection rules.",
                    "Document lessons learned and update playbooks.",
                ],
                "notes": [
                    "Steps are conditional due to limited confirmed details in the report."
                ],
            },
            "similar_incidents": retrieved,
            "assumptions": [
                "The report may be incomplete; environment and scope are not fully specified."
            ],
            "open_questions": [
                "What systems/accounts are affected?",
                "Are there confirmed indicators (IP/domain/hash/log entries)?",
                "What is the business impact and timeline?",
            ],
            "metadata": {"llm_provider": "mock", "model": self.model},
        }
