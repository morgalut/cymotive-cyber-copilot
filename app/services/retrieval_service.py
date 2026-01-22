import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

# FAISS is a library, not a service
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class KBIncident:
    id: str
    title: str
    description: str
    tags: List[str]
    mitigation_notes: List[str]


class RetrievalService:
    """
    Retrieval backends:
      - tfidf: TF-IDF cosine + optional BM25 hybrid
      - faiss: Embeddings + FAISS ANN + optional BM25 hybrid

    Notes:
      - For FAISS, we use cosine similarity by L2-normalizing vectors
        and using IndexFlatIP (inner product).
      - We use IndexIDMap2 so each vector is stored with a stable integer ID.
    """

    def __init__(self, incidents_path: str):
        self.incidents_path = incidents_path
        self.kb: List[KBIncident] = []

        # TF-IDF
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        # BM25
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus_tokens: List[List[str]] = []

        # FAISS
        self.faiss_index = None  # IndexIDMap2(IndexFlatIP)
        self.faiss_dim: Optional[int] = None
        self._faiss_intid_to_kb_idx: Dict[int, int] = {}  # int_id -> kb index
        self._kb_idx_to_faiss_intid: Dict[int, int] = {}  # kb index -> int_id

        # Embeddings client
        self._embed_client = None
        if self._backend() == "faiss":
            self._init_embed_client()


    # Public API

    def load_kb(self) -> None:
        with open(self.incidents_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.kb = [
            KBIncident(
                id=item["id"],
                title=item["title"],
                description=item["description"],
                tags=item.get("tags", []),
                mitigation_notes=item.get("mitigation_notes", []),
            )
            for item in raw
        ]

    def build_index(self) -> None:
        if not self.kb:
            raise RuntimeError("KB not loaded. Call load_kb() first.")

        texts = [self._to_doc_text(x) for x in self.kb]

        # Always build BM25 (needed for hybrid)
        self.bm25_corpus_tokens = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

        if self._backend() == "faiss":
            self._build_or_load_faiss(texts)
            return

        # TF-IDF backend
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(
        self, query: str, top_k: int = 2, use_hybrid: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.kb:
            raise RuntimeError("KB not loaded. Call load_kb().")

        t0 = time.time()
        backend = self._backend()

        if backend == "faiss":
            results = self._search_faiss(query, top_k=top_k, use_hybrid=use_hybrid)
            latency_ms = int((time.time() - t0) * 1000)
            meta = {
                "retrieval_latency_ms": latency_ms,
                "use_hybrid": use_hybrid,
                "top_k": top_k,
                "backend": "faiss",
            }
            return results, meta

        # tfidf backend
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF index not initialized. Call build_index().")

        q_vec = self.vectorizer.transform([query])
        cos_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        if use_hybrid and self.bm25 is not None:
            bm25_scores = np.array(self.bm25.get_scores(self._tokenize(query)), dtype=float)
            scores = self._fuse_scores_dense(cos_scores, bm25_scores, w_primary=0.55)
        else:
            scores = cos_scores

        results = self._format_topk(scores, top_k)
        latency_ms = int((time.time() - t0) * 1000)
        meta = {
            "retrieval_latency_ms": latency_ms,
            "use_hybrid": use_hybrid,
            "top_k": top_k,
            "backend": "tfidf",
        }
        return results, meta


    # Backend selection

    def _backend(self) -> str:
        return (getattr(settings, "RETRIEVAL_BACKEND", "tfidf") or "tfidf").lower().strip()


    # FAISS build/load

    def _init_embed_client(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed, but RETRIEVAL_BACKEND=faiss")
        if OpenAI is None:
            raise RuntimeError("openai package not installed (needed for embeddings)")
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing (needed for embeddings)")

        self._embed_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _build_or_load_faiss(self, texts: List[str]) -> None:
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)

        # Attempt load
        if self._faiss_files_exist():
            if self._try_load_faiss():
                return

        # Otherwise rebuild
        self._build_faiss_from_scratch(texts)

    def _faiss_files_exist(self) -> bool:
        return os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.FAISS_META_PATH)

    def _try_load_faiss(self) -> bool:
        try:
            with open(settings.FAISS_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

            dim = int(meta["dim"])
            id_map: List[str] = list(meta["id_map"])  # list of KB incident string IDs in kb order

            index = faiss.read_index(settings.FAISS_INDEX_PATH)

            # Validate basic consistency
            if index.d != dim:
                return False
            if len(id_map) != len(self.kb):
                return False

            self.faiss_dim = dim
            self.faiss_index = index

            # Rebuild int-id mapping deterministically from current KB order
            # Each vector stored uses int IDs (kb_idx)
            self._faiss_intid_to_kb_idx = {i: i for i in range(len(self.kb))}
            self._kb_idx_to_faiss_intid = {i: i for i in range(len(self.kb))}
            return True

        except Exception:
            return False

    def _build_faiss_from_scratch(self, texts: List[str]) -> None:
        if self._embed_client is None:
            self._init_embed_client()

        # Embed docs
        emb = self._embed_texts(texts)  # (N, dim) float32
        if emb.ndim != 2:
            raise RuntimeError("Embeddings shape invalid")

        n, dim = emb.shape
        self.faiss_dim = dim

        # Normalize for cosine similarity, then use inner product
        faiss.normalize_L2(emb)

        # IndexFlatIP is exact (good for small KB), IndexIDMap2 adds stable IDs
        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base)

        # Use integer IDs == kb index (stable if KB ordering stable)
        # If you later want full stability across KB reorders, store a persistent mapping.
        ids = np.arange(n).astype(np.int64)
        index.add_with_ids(emb, ids)

        self.faiss_index = index
        self._faiss_intid_to_kb_idx = {int(i): int(i) for i in ids}
        self._kb_idx_to_faiss_intid = {int(i): int(i) for i in ids}

        # Persist index + meta
        faiss.write_index(index, settings.FAISS_INDEX_PATH)
        with open(settings.FAISS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": dim,
                    # store the KB string ids in order (useful sanity check)
                    "id_map": [inc.id for inc in self.kb],
                },
                f,
            )


    # FAISS search

    def _search_faiss(self, query: str, top_k: int, use_hybrid: bool) -> List[Dict[str, Any]]:
        if self.faiss_index is None or self.faiss_dim is None:
            raise RuntimeError("FAISS index not initialized. Call build_index().")

        q = self._embed_texts([query])  # (1, dim)
        faiss.normalize_L2(q)

        scores, ids = self.faiss_index.search(q, top_k)
        faiss_scores = scores.flatten()
        faiss_int_ids = ids.flatten()

        # Dense vector of size N for hybrid fusion
        dense = np.zeros((len(self.kb),), dtype=float)

        for s, int_id in zip(faiss_scores, faiss_int_ids):
            if int_id < 0:
                continue
            kb_idx = self._faiss_intid_to_kb_idx.get(int(int_id))
            if kb_idx is None:
                continue
            dense[kb_idx] = float(s)

        if use_hybrid and self.bm25 is not None:
            bm25_scores = np.array(self.bm25.get_scores(self._tokenize(query)), dtype=float)
            scores_final = self._fuse_scores_dense(dense, bm25_scores, w_primary=0.65)
        else:
            scores_final = dense

        return self._format_topk(scores_final, top_k)


    # Embeddings

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self._embed_client is None:
            raise RuntimeError("Embeddings client not initialized")

        resp = self._embed_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


    # Score fusion helpers

    def _fuse_scores_dense(self, primary: np.ndarray, secondary: np.ndarray, w_primary: float) -> np.ndarray:
        """
        Min-max normalize both arrays and fuse:
          score = w_primary * norm(primary) + (1-w_primary) * norm(secondary)
        """
        primary_norm = self._safe_minmax(primary.astype(float))
        secondary_norm = self._safe_minmax(secondary.astype(float))
        return w_primary * primary_norm + (1.0 - w_primary) * secondary_norm


    # Shared formatting/helpers

    def _format_topk(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for i in top_idx:
            inc = self.kb[int(i)]
            results.append(
                {
                    "id": inc.id,
                    "title": inc.title,
                    "snippet": self._make_snippet(inc.description),
                    "score": float(scores[int(i)]),
                    "match_factors": inc.tags[:5],
                }
            )
        return results

    def _to_doc_text(self, inc: KBIncident) -> str:
        tags = " ".join(inc.tags)
        mit = " ".join(inc.mitigation_notes)
        return f"{inc.title}\n{inc.description}\nTAGS: {tags}\nMITIGATION: {mit}"

    def _make_snippet(self, text: str, max_len: int = 240) -> str:
        text = " ".join(text.split())
        return text[:max_len] + ("..." if len(text) > max_len else "")

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if t]

    def _safe_minmax(self, arr: np.ndarray) -> np.ndarray:
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-9:
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)
