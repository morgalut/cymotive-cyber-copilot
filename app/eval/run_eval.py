"""PoC evaluation runner.

Creates a small, repeatable evaluation loop over `golden_inputs.json`.

It supports two execution modes:
1) **API mode (recommended)**: call a running FastAPI instance.
   - Set environment variable: EVAL_BASE_URL (default: http://127.0.0.1:8000)
   - Endpoint expected: POST /v1/incidents/analyze

2) **Local fallback mode**: if the API is not reachable, run a minimal in-script
   "mock copilot" so the evaluator can still run end-to-end.

This is intentionally a *heuristic* evaluator suitable for a take-home PoC, not a
production benchmark.

Run:
  python app/eval/run_eval.py

Optional env vars:
  EVAL_BASE_URL=http://127.0.0.1:8000
  EVAL_TIMEOUT_S=30
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


HERE = os.path.dirname(__file__)
GOLDEN_PATH = os.path.join(HERE, "golden_inputs.json")

BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_S = int(os.getenv("EVAL_TIMEOUT_S", "30"))


# ----------------------------
# Utilities
# ----------------------------

def _contains_any(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return any(n.lower() in t for n in needles)


def _list_contains_any(items: List[str], needles: List[str]) -> bool:
    joined = " ".join(items or []).lower()
    return any(n.lower() in joined for n in needles)


def _safe_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _is_jsonable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False


# ----------------------------
# API client (preferred)
# ----------------------------

def call_analyze_api(report_text: str, top_k: int = 2, use_hybrid: bool = True) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed; cannot run API mode")

    url = f"{BASE_URL}/v1/incidents/analyze"
    payload = {
        "report_text": report_text,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "response_style": "standard",
    }
    r = requests.post(url, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def api_available() -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ----------------------------
# Local fallback "mock" copilot
# ----------------------------

def local_fallback_copilot(report_text: str) -> Dict[str, Any]:
    """A minimal, deterministic copilot output (used only if API isn't reachable).

    This is NOT the main deliverable; it's here so evaluation tooling can run
    even before wiring a real model.
    """
    txt = report_text.strip()

    # crude category guess from keywords (for evaluation plumbing only)
    lowered = txt.lower()
    if any(k in lowered for k in ["ransom", "extension", "encrypt"]):
        cat = "ransomware"
    elif any(k in lowered for k in ["phish", "sso", "mfa"]):
        cat = "phishing"
    elif any(k in lowered for k in ["ddos", "latency", "traffic spike", "waf"]):
        cat = "ddos"
    elif "powershell" in lowered:
        cat = "malware"
    else:
        cat = None

    open_q = [
        "Which systems/accounts are affected?",
        "Are there confirmed IOCs (IP/domain/hash/log lines)?",
        "What is the timeline and business impact?",
    ]
    if len(txt) < 80:
        # edge-case: very messy
        open_q.append("What security tooling alerts/logs are available (EDR/AV/SIEM)?")

    return {
        "summary": "Preliminary summary: limited confirmed details in the report; requires triage.",
        "key_indicators": [],
        "suspected_category": cat,
        "mitigation_plan": {
            "immediate": [
                "Contain suspected affected assets (isolate host/server if confirmed compromised).",
                "Preserve logs and volatile evidence; avoid wiping until scope is known.",
            ],
            "short_term": [
                "Validate scope and entry point; review auth, EDR, and network telemetry.",
                "Identify and remove persistence; reset/disable compromised identities as needed.",
            ],
            "long_term": [
                "Harden access controls (MFA, least privilege) and patch exposed services.",
                "Update detections and playbooks based on lessons learned.",
            ],
            "notes": ["Actions are conditional due to incomplete input."],
        },
        "similar_incidents": [],
        "assumptions": ["Report is incomplete; environment details are missing."],
        "open_questions": open_q,
        "metadata": {"mode": "local_fallback"},
    }


# ----------------------------
# Heuristic scoring
# ----------------------------

@dataclass
class CaseScore:
    case_id: str
    title: str
    score_total: int
    score_breakdown: Dict[str, int]
    notes: List[str]


def score_case(output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[int, Dict[str, int], List[str]]:
    notes: List[str] = []
    breakdown: Dict[str, int] = {}

    # A) Validity & structure (0-2)
    valid = 2
    if not isinstance(output, dict) or not _is_jsonable(output):
        valid = 0
        notes.append("Output is not JSON-serializable dict")
    else:
        required_paths = [
            "summary",
            "mitigation_plan.immediate",
            "mitigation_plan.short_term",
            "mitigation_plan.long_term",
            "similar_incidents",
            "open_questions",
        ]
        missing = [p for p in required_paths if _safe_get(output, p, None) is None]
        if missing:
            valid = 1
            notes.append(f"Missing required fields: {missing}")
    breakdown["structure"] = valid

    # B) Category match (0-2)
    cat_score = 0
    out_cat = (_safe_get(output, "suspected_category", None) or "").lower()
    expected_cats = expected.get("suspected_category_any_of", [])
    if not expected_cats:
        cat_score = 2
    else:
        if out_cat and any(ec.lower() in out_cat for ec in expected_cats):
            cat_score = 2
        elif out_cat:
            cat_score = 1
            notes.append(f"Category mismatch: got '{out_cat}', expected any of {expected_cats}")
        else:
            cat_score = 0
            notes.append("Missing suspected_category")
    breakdown["category"] = cat_score

    # C) Summary contains key signals (0-2)
    summary_score = 0
    out_summary = _safe_get(output, "summary", "") or ""
    must_any = expected.get("summary_must_contain_any_of", [])
    if must_any:
        if _contains_any(out_summary, must_any):
            summary_score = 2
        else:
            summary_score = 0
            notes.append(f"Summary missing key terms (any of): {must_any}")
    else:
        summary_score = 2
    breakdown["summary"] = summary_score

    # D) Immediate mitigation actionability (0-2)
    mit_score = 0
    immediate = _safe_get(output, "mitigation_plan.immediate", []) or []
    must_any = expected.get("mitigation_immediate_must_contain_any_of", [])
    if must_any:
        if _list_contains_any(immediate, must_any):
            mit_score = 2
        elif immediate:
            mit_score = 1
            notes.append(f"Immediate steps present but missing key terms (any of): {must_any}")
        else:
            mit_score = 0
            notes.append("No immediate mitigation steps")
    else:
        mit_score = 2
    breakdown["mitigation_immediate"] = mit_score

    # E) Open questions minimum (0-2)
    oq_score = 0
    oq = _safe_get(output, "open_questions", []) or []
    min_q = int(expected.get("open_questions_min", 0))
    if len(oq) >= min_q:
        oq_score = 2
    elif len(oq) > 0:
        oq_score = 1
        notes.append(f"Open questions count {len(oq)} < required {min_q}")
    else:
        oq_score = 0
        notes.append("No open_questions")
    breakdown["open_questions"] = oq_score

    # F) Similar incidents presence (0-2)
    sim_score = 0
    sims = _safe_get(output, "similar_incidents", []) or []
    if isinstance(sims, list) and len(sims) >= 1:
        sim_score = 2
    elif isinstance(sims, list):
        sim_score = 1
        notes.append("No similar_incidents returned (may be OK early PoC, but hurts RAG grounding)")
    else:
        sim_score = 0
        notes.append("similar_incidents not a list")
    breakdown["similar_incidents"] = sim_score

    total = sum(breakdown.values())
    return total, breakdown, notes


def main() -> int:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    use_api = api_available()
    mode = "API" if use_api else "LOCAL_FALLBACK"

    print(f"\n== PoC Eval Runner ==")
    print(f"Mode: {mode}")
    if use_api:
        print(f"Base URL: {BASE_URL}")
    else:
        print("API not reachable; using local fallback output generator.")
        print("Tip: run your FastAPI server and set EVAL_BASE_URL if needed.")

    results: List[CaseScore] = []
    grand_total = 0
    max_total = 0

    for case in golden:
        cid = case["id"]
        title = case.get("title", "")
        report_text = case["report_text"]
        expected = case.get("expected", {})

        t0 = time.time()
        if use_api:
            out = call_analyze_api(report_text)
        else:
            out = local_fallback_copilot(report_text)
        latency_ms = int((time.time() - t0) * 1000)

        total, breakdown, notes = score_case(out, expected)
        notes.append(f"latency_ms={latency_ms}")

        results.append(CaseScore(cid, title, total, breakdown, notes))

        grand_total += total
        max_total += 12  # 6 dimensions x (0-2)

    print("\n== Results ==")
    for r in results:
        print(f"\n[{r.case_id}] {r.title}")
        print(f"  Score: {r.score_total}/12")
        print(f"  Breakdown: {r.score_breakdown}")
        for n in r.notes:
            print(f"  - {n}")

    pct = (grand_total / max_total * 100.0) if max_total else 0.0
    print("\n== Summary ==")
    print(f"Total: {grand_total}/{max_total} ({pct:.1f}%)")

    # Non-zero exit if very low score (useful for CI)
    return 0 if pct >= 60.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
