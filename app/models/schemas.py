# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class IncidentRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Free-text incident report or query")
    top_k: int = Field(2, ge=1, le=5)
    use_hybrid: bool = Field(False, description="Bonus: combine keyword + semantic")

class RetrievedIncident(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    match_factors: List[str] = Field(default_factory=list, description="Why this matched (e.g., ransomware, AD, VPN)")

class IncidentRetrieveResponse(BaseModel):
    matches: List[RetrievedIncident]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IncidentAnalyzeRequest(BaseModel):
    report_text: str = Field(..., min_length=20)
    top_k: int = Field(2, ge=1, le=5)
    use_hybrid: bool = False
    response_style: Literal["concise", "standard", "detailed"] = "standard"

class MitigationPlan(BaseModel):
    immediate: List[str] = Field(default_factory=list, description="Containment actions within minutes-hours")
    short_term: List[str] = Field(default_factory=list, description="Eradication + stabilization steps")
    long_term: List[str] = Field(default_factory=list, description="Hardening + prevention steps")
    notes: List[str] = Field(default_factory=list, description="Caveats, dependencies, environment constraints")

class IncidentAnalyzeResponse(BaseModel):
    summary: str
    key_indicators: List[str] = Field(default_factory=list, description="IPs, hashes, domains, logs, artifacts (if present)")
    suspected_category: Optional[str] = Field(None, description="e.g., phishing, malware, ransomware, DDoS, insider")
    mitigation_plan: MitigationPlan
    similar_incidents: List[RetrievedIncident] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="model, latency_ms, token_estimate, retrieval_scores")
