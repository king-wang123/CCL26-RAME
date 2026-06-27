"""Skill retriever: embedding-based retrieval of relevant extraction skills.

Loads deduplicated skills + pre-computed embeddings, encodes query text,
returns top-K most relevant skills by cosine similarity.

Usage:
  from skill_retriever import SkillRetriever
  retriever = SkillRetriever("data/skills_dedup.json", "data/skills_embeddings.npy")
  skills = retriever.retrieve("Some crop breeding text...", top_k=10)
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/all-mpnet-base-v2",
)


class SkillRetriever:
    """Retrieve relevant extraction skills by embedding cosine similarity."""

    def __init__(
        self,
        skills_json: str = "data/skills_dedup.json",
        embeddings_npy: str = "data/skills_embeddings.npy",
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
    ):
        # Load skills
        with open(skills_json) as f:
            self.skills = json.load(f)
        self.skill_texts = [s["skill"] for s in self.skills]

        # Load pre-computed embeddings
        self.skill_embeddings = np.load(embeddings_npy)
        assert self.skill_embeddings.shape[0] == len(self.skills), (
            f"Embedding count {self.skill_embeddings.shape[0]} != skill count {len(self.skills)}"
        )

        # Load model for query encoding
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        print(f"[SkillRetriever] Loaded {len(self.skills)} skills, embedding model on {device}", flush=True)

    @torch.no_grad()
    def _encode_query(self, text: str) -> np.ndarray:
        """Encode query text into L2-normalized embedding."""
        encoded = self.tokenizer(
            [text], padding=True, truncation=True, max_length=384, return_tensors="pt"
        ).to(self.device)
        output = self.model(**encoded)
        attention_mask = encoded["attention_mask"]
        token_embeddings = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve top-K most relevant skill texts for the given query."""
        q_emb = self._encode_query(query)
        scores = self.skill_embeddings @ q_emb
        top_indices = np.argsort(-scores)[:top_k]
        return [self.skill_texts[i] for i in top_indices]

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[tuple]:
        """Retrieve top-K skills with similarity scores."""
        q_emb = self._encode_query(query)
        scores = self.skill_embeddings @ q_emb
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.skill_texts[i], float(scores[i])) for i in top_indices]

    def retrieve_by_category(self, query: str, top_k: int = 10, categories: Optional[List[str]] = None) -> List[str]:
        """Retrieve top-K skills, optionally filtered by category."""
        if categories is None:
            return self.retrieve(query, top_k)

        q_emb = self._encode_query(query)
        scores = self.skill_embeddings @ q_emb

        valid_indices = [i for i, s in enumerate(self.skills) if s.get("category") in categories]
        if not valid_indices:
            return self.retrieve(query, top_k)

        valid_scores = [(scores[i], i) for i in valid_indices]
        valid_scores.sort(reverse=True)
        top_indices = [idx for _, idx in valid_scores[:top_k]]
        return [self.skill_texts[i] for i in top_indices]
