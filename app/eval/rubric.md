# Evaluation Rubric (PoC)

This rubric is designed to evaluate the PoC outputs for **summary**, **mitigation plan**, and **retrieval relevance** in a way that aligns with the assignment goals (actionability, hallucination-risk awareness, and operational usefulness).

## Scoring scale
Each dimension is scored **0–2**:
- **0** = missing / incorrect / unsafe
- **1** = partial / generic / some issues
- **2** = correct, clear, actionable, and appropriately cautious

Overall score is the sum across dimensions.

---

## A) Output validity & structure (0–2)
**Checks**
- Output parses as JSON (or structured fields if not JSON).
- Contains required sections: `summary`, `mitigation_plan.immediate/short_term/long_term`, `similar_incidents`.

**2**: All required fields present and well-structured.

---

## B) Summary correctness (0–2)
**Checks**
- Captures the *core incident story*: what happened, which asset(s), key symptoms, and time signals if present.
- Does **not** invent IOCs, affected systems, or root cause.

**2**: Accurate, concise, no invented facts.

---

## C) Mitigation actionability & ordering (0–2)
**Checks**
- Steps are prioritized: **containment → eradication → recovery → prevention**.
- Steps are concrete (isolate host, disable account, revoke sessions, rate limiting, etc.), not just "investigate".

**2**: Clear and operational with sensible sequencing.

---

## D) Grounding & hallucination control (0–2)
**Checks**
- Assumptions are explicitly labeled.
- Missing details become **open questions**.
- Similar-incidents section references only retrieved items.

**2**: Conservative language + clear uncertainties + grounded retrieval use.

---

## E) Retrieval relevance (0–2)
**Checks**
- Retrieved incidents plausibly match the report category/TTPs (e.g., ransomware ↔ ransomware; phishing ↔ identity).
- Top matches are not obviously unrelated.

**2**: Top-2 are strongly relevant.

---

## F) Analyst usefulness (0–2)
**Checks**
- Output is skimmable.
- Contains the next best actions an analyst can take.
- Avoids overly long generic checklists.

**2**: Immediately useful during triage.

---

## Notes for the edge-case input
For noisy/incomplete reports, the correct behavior is:
- Short, cautious summary
- Increased number of open questions
- Conditional mitigation actions

