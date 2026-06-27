"""Embedding-based retriever using sentence-transformers for semantic similarity.

Provides:
  - EmbeddingRetriever: pure semantic retrieval via cosine similarity
  - HybridRetriever: score fusion of BM25 + embedding retrieval via RRF

Supports pre-computed embeddings via .npy cache for fast startup:
  # Pre-compute:
  python embedding_retriever.py --docs data/train.json --output data/train_embeddings.npy
  # Load in predict.py:
  EmbeddingRetriever(docs, cache_path="data/train_embeddings.npy")
"""
from __future__ import annotations
import os
import numpy as np
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/all-mpnet-base-v2",
)


def _mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class EmbeddingRetriever:
    """Semantic retriever using all-mpnet-base-v2 (768-dim, mean pooling, L2-normalized).

    Encodes all pool documents at init time, then retrieves top-M by cosine similarity.
    """

    def __init__(
        self,
        docs: List[str],
        model_path: str = DEFAULT_MODEL_PATH,
        batch_size: int = 64,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        self.docs = docs
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Try loading pre-computed embeddings from cache
        if cache_path and os.path.exists(cache_path):
            print(f"[EmbeddingRetriever] loading pre-computed embeddings from {cache_path}...", flush=True)
            self.doc_embeddings = np.load(cache_path)
            assert self.doc_embeddings.shape[0] == len(docs), (
                f"Cache size mismatch: {self.doc_embeddings.shape[0]} vs {len(docs)} docs"
            )
            print(f"[EmbeddingRetriever] loaded cache. Shape: {self.doc_embeddings.shape}", flush=True)
            # Still need model for query encoding
            print(f"[EmbeddingRetriever] loading model for query encoding...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(device).eval()
            print(f"[EmbeddingRetriever] ready (cached mode).", flush=True)
        else:
            print(f"[EmbeddingRetriever] loading model from {model_path} on {device}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(device).eval()

            print(f"[EmbeddingRetriever] encoding {len(docs)} pool documents...", flush=True)
            self.doc_embeddings = self._encode_batch(docs, batch_size)  # (N, 768) numpy
            print(f"[EmbeddingRetriever] ready. Embedding shape: {self.doc_embeddings.shape}", flush=True)

            # Auto-save cache if cache_path specified but didn't exist
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.save(cache_path, self.doc_embeddings)
                print(f"[EmbeddingRetriever] saved embeddings to {cache_path}", flush=True)

    @torch.no_grad()
    def _encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into L2-normalized embeddings."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=384, return_tensors="pt"
            ).to(self.device)
            output = self.model(**encoded)
            embs = _mean_pooling(output, encoded["attention_mask"])
            # L2 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into L2-normalized embedding."""
        return self._encode_batch([query], batch_size=1)[0]  # (768,)

    def top_m(self, query: str, m: int) -> List[int]:
        """Return indices of top-m most similar documents by cosine similarity."""
        q_emb = self._encode_query(query)  # (768,)
        # Since both are L2-normalized, cosine sim = dot product
        scores = self.doc_embeddings @ q_emb  # (N,)
        top_indices = np.argsort(-scores)[:m].tolist()
        return top_indices

    def top_m_with_scores(self, query: str, m: int) -> List[tuple]:
        """Return (index, score) pairs for top-m documents."""
        q_emb = self._encode_query(query)
        scores = self.doc_embeddings @ q_emb
        top_indices = np.argsort(-scores)[:m]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class HybridRetriever:
    """Hybrid retriever combining BM25 and embedding scores via Reciprocal Rank Fusion (RRF).

    RRF score for doc d = sum over retrievers r of: 1 / (k + rank_r(d))
    where k is a constant (default 60).
    """

    def __init__(self, bm25_retriever, embedding_retriever: EmbeddingRetriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.emb = embedding_retriever
        self.rrf_k = rrf_k

    def top_m(self, query: str, m: int) -> List[int]:
        """Return top-m indices using RRF fusion of BM25 and embedding rankings."""
        candidate_m = min(m * 3, len(self.emb.docs))

        bm25_ranking = self.bm25.top_m(query, candidate_m)
        emb_ranking = self.emb.top_m(query, candidate_m)

        # Compute RRF scores
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(emb_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score descending
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        return sorted_candidates[:m]

    def top_m_with_scores(self, query: str, m: int) -> List[tuple]:
        """Return (index, rrf_score) pairs for top-m documents."""
        candidate_m = min(m * 3, len(self.emb.docs))

        bm25_ranking = self.bm25.top_m(query, candidate_m)
        emb_ranking = self.emb.top_m(query, candidate_m)

        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(emb_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_candidates = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:m]
        return [(idx, rrf_scores[idx]) for idx in sorted_candidates]


# ── CLI: pre-compute and save embeddings ──────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Pre-compute document embeddings and save as .npy")
    p.add_argument("--docs", required=True, help="Input JSON file (e.g. data/train.json)")
    p.add_argument("--output", required=True, help="Output .npy file (e.g. data/train_embeddings.npy)")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Embedding model path")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    print(f"[precompute] Loading docs from {args.docs}...", flush=True)
    docs = json.load(open(args.docs))
    texts = [d["text"] for d in docs]
    print(f"[precompute] {len(texts)} documents loaded.", flush=True)

    retriever = EmbeddingRetriever(texts, model_path=args.model, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, retriever.doc_embeddings)
    print(f"[precompute] Saved embeddings ({retriever.doc_embeddings.shape}) -> {args.output}", flush=True)
