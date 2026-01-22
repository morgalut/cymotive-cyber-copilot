Below is a **ready-to-drop `README.md`** that documents **all CURL tests**, explains **what each test validates**, and shows **how to run the evaluation (eval/VAL) script**.
It is written to match *exactly* what your system currently does and what the assignment expects to see.

You can copy-paste this into `README.md` at the repo root.

---

# Cymotive Cyber Copilot – PoC README

## Overview

This project implements a **GenAI-powered Cybersecurity Incident Copilot**.
It accepts free-text incident reports, retrieves similar historical incidents (RAG), and generates a **structured, grounded analysis** including:

* Incident summary
* Suspected category
* Key indicators
* Incident-specific mitigation plan
* Assumptions and open questions
* Observability metadata (latency, tokens, model, retrieval info)

The system is exposed via a **FastAPI service** and supports **hybrid retrieval (TF-IDF + BM25)** and **LLM structured outputs**.

---

## Running the API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Expected output:

```json
{"status":"ok"}
```

---

## API Endpoints

### 1. Analyze Incident (`/v1/incidents/analyze`)

This is the main GenAI endpoint.
It performs:

1. Retrieval of similar incidents (RAG)
2. LLM-based summarization + mitigation
3. Grounding enforcement
4. Metadata enrichment (latency, tokens, model)

---

## CURL Test Cases

### Test 1 – Ransomware-like incident (hybrid retrieval ON)

**Purpose:**
Validates end-to-end flow: retrieval + LLM + grounding + mitigation specificity.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/incidents/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "report_text": "We detected rapid file modifications on the finance file server. Several files now have strange extensions and a ransom note appeared in multiple folders. One admin account showed unusual logins overnight. Backups status unknown.",
    "top_k": 2,
    "use_hybrid": true,
    "response_style": "standard"
  }'
```

**What this test demonstrates:**

* Correct categorization as `ransomware`
* Retrieval of relevant incidents (INC-003, INC-006)
* Incident-specific mitigation steps
* Grounded output (no hallucinated incidents)
* Token usage + latency metadata

---

### Test 2 – Phishing + MFA fatigue scenario

**Purpose:**
Validates identity-focused incidents and hybrid retrieval behavior.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/incidents/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "report_text": "Multiple users received an email mimicking our SSO page. One user entered credentials. After that, we saw repeated MFA prompts and then a successful login from an unusual location.",
    "top_k": 2,
    "use_hybrid": true,
    "response_style": "standard"
  }'
```

**What this test demonstrates:**

* Phishing / account-compromise detection
* Identity-centric mitigation actions
* Retrieval relevance for identity incidents

---

### Test 3 – Noisy / incomplete report (edge case)

**Purpose:**
Validates robustness to low-quality input.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/incidents/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "report_text": "weird stuff again... lots of alerts. maybe vpn? user said popups. idk.",
    "top_k": 2,
    "use_hybrid": true,
    "response_style": "concise"
  }'
```

**What this test demonstrates:**

* Conservative summarization
* Use of `assumptions` and `open_questions`
* No hallucination of facts

---

### Test 4 – Input validation (error handling)

**Purpose:**
Ensures schema validation is enforced before LLM execution.

```bash
curl -i -X POST "http://127.0.0.1:8000/v1/incidents/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "report_text": "too short",
    "top_k": 2,
    "use_hybrid": false,
    "response_style": "standard"
  }'
```

**Expected result:**
HTTP `422 Unprocessable Entity`

This confirms:

* Validation happens early
* Invalid input does not trigger LLM calls (cost control)

---

### 5. Retrieve Similar Incidents Only (`/v1/incidents/retrieve`)

**Purpose:**
Tests retrieval independently from the LLM.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/incidents/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ransom note file server rapid encryption admin compromise",
    "top_k": 2,
    "use_hybrid": true
  }' | jq .
```

**What this test demonstrates:**

* Retrieval quality
* Hybrid ranking behavior
* Transparent similarity scores

---

## Running the Evaluation (VAL / Eval Harness)

The project includes a lightweight evaluation harness with **golden inputs** and a **scoring rubric**.

### Files

* `app/eval/golden_inputs.json` – curated test cases
* `app/eval/rubric.md` – scoring criteria
* `app/eval/run_eval.py` – evaluation runner

### Run evaluation

```bash
python app/eval/run_eval.py
```

**What the evaluation checks:**

* Output structure correctness
* Summary relevance
* Mitigation actionability
* Retrieval grounding
* Hallucination risk
* Latency visibility

If the API is not running, the script falls back to a local mock so the harness always executes.

---

## Observability & Monitoring

Each `/analyze` response includes a `metadata` block:

```json
{
  "latency_ms": 10936,
  "llm_provider": "openai",
  "model": "gpt-4o-mini",
  "usage": {
    "total_tokens": 1301
  },
  "retrieval": {
    "use_hybrid": true,
    "top_k": 2,
    "retrieval_latency_ms": 2
  },
  "request_id": "uuid"
}
```

This supports:

* Cost awareness
* Performance tracking
* Debugging & evaluation

---

## Summary

This PoC demonstrates:

* A working GenAI + RAG pipeline
* Grounded, structured LLM outputs
* Clear separation of retrieval and generation
* Robust edge-case handling
* Evaluation and monitoring hooks

The architecture is intentionally modular to support future improvements such as:

* Embedding-based retrieval
* Reranking
* Automated evaluations
* Confidence scoring and self-checks


