"""Deduplicate distilled skills using embedding similarity.

Load raw skills from JSONL, encode with sentence-transformers, greedy deduplicate
by cosine similarity threshold, output deduplicated skill list + embeddings.

Usage:
  python deduplicate_skills.py --input data/skills_raw.jsonl \
      --output data/skills_dedup.json --embeddings data/skills_embeddings.npy
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from typing import List, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_raw_skills(path: str) -> List[Dict]:
    """Load all skills from JSONL, flatten into a list of skill dicts."""
    all_skills = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            for s in obj.get("skills", []):
                all_skills.append({
                    "skill": s["skill"],
                    "category": s.get("category", "unknown"),
                    "source_doc_idx": obj["doc_idx"],
                })
    return all_skills


def encode_skills(skills: List[str], model_path: str, batch_size: int = 64, device: str = None) -> np.ndarray:
    """Encode skill texts into L2-normalized embeddings."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[dedup] Loading embedding model from {model_path} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(skills), batch_size):
            batch = skills[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            output = model(**encoded)
            # Mean pooling
            attention_mask = encoded["attention_mask"]
            token_embeddings = output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # L2 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  Encoded {min(i + batch_size, len(skills))}/{len(skills)}", flush=True)

    return np.vstack(all_embs)


def greedy_dedup(embeddings: np.ndarray, threshold: float = 0.85) -> List[int]:
    """Greedy deduplication: iterate skills in order, skip if too similar to any kept skill."""
    n = len(embeddings)
    kept_indices = []
    kept_embs = []

    for i in range(n):
        if kept_embs:
            sims = np.dot(np.array(kept_embs), embeddings[i])
            if np.max(sims) >= threshold:
                continue
        kept_indices.append(i)
        kept_embs.append(embeddings[i])

    return kept_indices


def run(args):
    print(f"[dedup] Loading raw skills from {args.input}...", flush=True)
    all_skills = load_raw_skills(args.input)
    print(f"[dedup] Loaded {len(all_skills)} raw skills", flush=True)

    if not all_skills:
        print("[dedup] No skills to deduplicate!", flush=True)
        return

    # Encode
    skill_texts = [s["skill"] for s in all_skills]
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2")
    embeddings = encode_skills(skill_texts, model_path, batch_size=args.batch_size)
    print(f"[dedup] Embeddings shape: {embeddings.shape}", flush=True)

    # Deduplicate
    print(f"[dedup] Deduplicating with threshold={args.threshold}...", flush=True)
    kept_idx = greedy_dedup(embeddings, threshold=args.threshold)
    print(f"[dedup] Kept {len(kept_idx)}/{len(all_skills)} skills ({100*len(kept_idx)/len(all_skills):.1f}%)", flush=True)

    # Build output
    dedup_skills = [all_skills[i] for i in kept_idx]
    dedup_embeddings = embeddings[kept_idx]

    # Category distribution
    from collections import Counter
    cat_dist = Counter(s["category"] for s in dedup_skills)
    print(f"[dedup] Category distribution: {dict(cat_dist)}", flush=True)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dedup_skills, f, ensure_ascii=False, indent=2)
    print(f"[dedup] Saved {len(dedup_skills)} skills -> {args.output}", flush=True)

    np.save(args.embeddings, dedup_embeddings)
    print(f"[dedup] Saved embeddings ({dedup_embeddings.shape}) -> {args.embeddings}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Deduplicate distilled skills via embedding similarity")
    p.add_argument("--input", default="data/skills_raw.jsonl", help="Raw skills JSONL")
    p.add_argument("--output", default="data/skills_dedup.json", help="Output deduplicated JSON")
    p.add_argument("--embeddings", default="data/skills_embeddings.npy", help="Output embeddings npy")
    p.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold for dedup")
    p.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
