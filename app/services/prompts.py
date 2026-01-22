# app/services/prompts.py

SYSTEM_PROMPT = """You are a cybersecurity incident response copilot assisting SOC analysts.
Be precise, cautious, and operational. Do not invent facts.
If information is missing, state assumptions clearly and list open questions.
When referencing "similar incidents", ONLY use the provided retrieved incidents and cite their IDs (e.g., INC-003).
Output MUST be valid JSON and MUST follow the provided schema.
"""

# Prompt 1: focused summarization + categorization + indicators extraction
SUMMARY_PROMPT = """Analyze the incident report and produce a compact, factual summary.

REQUIREMENTS:
- Summarize only what is explicitly stated.
- Extract key indicators (IOCs/artifacts) ONLY if explicitly present in the report text.
- Assign suspected_category from: ["ransomware","phishing","malware","ddos","account_compromise","dns_tunneling","unknown"].
- If the report is noisy/incomplete, keep summary minimal and add missing details to open_questions.

INCIDENT REPORT:
{report_text}

Return ONLY JSON with keys:
- summary (string)
- suspected_category (string)
- key_indicators (array of strings)
- assumptions (array of strings)
- open_questions (array of strings)
"""

# Prompt 2: mitigation plan grounded in report + retrieved incidents
MITIGATION_PROMPT = """You will be given:
1) Incident report (free-text)
2) Retrieved similar incidents (a list of objects with fields id,title,description/tags/mitigation_notes)

TASK:
Create an incident response mitigation plan tailored to the report.

RULES:
- Do NOT invent environment details.
- Prioritize actions: containment -> eradication -> recovery -> prevention.
- Use retrieved incidents ONLY as optional guidance and cite their IDs when you borrow ideas (e.g., "Based on INC-003, consider...").
- If uncertain, phrase actions conditionally and list open questions.

INCIDENT REPORT:
{report_text}

RETRIEVED SIMILAR INCIDENTS:
{retrieved_incidents_json}

Return ONLY JSON with keys:
- mitigation_plan: {{ "immediate": [], "short_term": [], "long_term": [], "notes": [] }}
"""

# One-call combined prompt (optional). If you prefer a single LLM call, use this.
COMBINED_ANALYZE_PROMPT = """You will be given:
1) Incident report (free-text)
2) Retrieved similar incidents (knowledge base hits)

TASK:
Produce a JSON object with fields:
- summary (string; must mention the main incident type and key facts from the report)
- key_indicators (list; only explicitly present indicators)
- suspected_category (one of ["ransomware","phishing","malware","ddos","account_compromise","dns_tunneling","unknown"])
- mitigation_plan: {{ "immediate": [], "short_term": [], "long_term": [], "notes": [] }}
- similar_incidents: (reuse the retrieved incidents EXACTLY; do NOT add new ones)
- assumptions (list)
- open_questions (list)

GROUNDING RULES:
- Do not guess missing facts.
- When mitigation uses ideas from retrieved incidents, cite incident IDs inside mitigation_plan.notes.
- If retrieval appears weak (low similarity), say so in notes.

INCIDENT REPORT:
{report_text}

RETRIEVED SIMILAR INCIDENTS:
{retrieved_incidents_json}

RESPONSE_STYLE: {response_style}
Return ONLY valid JSON.
"""
