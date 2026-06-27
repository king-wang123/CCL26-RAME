"""Skill-augmented MGBIE predictor: V6+V7+V8 ensemble with skill retrieval.

Instead of retrieving raw training documents as few-shot demos, this version
retrieves fine-grained extraction skills and injects them as "Extraction Guidelines"
into the system prompt. A small number of demos (K_DEMO=4) are still used for
output format reference.

Usage:
  # Inference on test set:
  python predict_skill.py --input data/test.json --output results/submit_skill.json \
      --pool data/train.json --skills data/skills_dedup.json \
      --skill_embeddings data/skills_embeddings.npy --workers 16

  # Test on a few docs:
  python predict_skill.py --input data/test.json --output results/test_skill.json \
      --pool data/train.json --skills data/skills_dedup.json \
      --skill_embeddings data/skills_embeddings.npy --test 3
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, sys, time, random
from typing import Any, Dict, List, Optional
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import LLM
from retriever import BM25Retriever
from skill_retriever import SkillRetriever
from prompts_skill import (
    V6_SKILL_BASE, V7_SKILL_BASE, V8_SKILL_BASE,
    build_messages_with_skills,
)
from tools import extract_json_obj, to_official, majority_vote

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_DRAWS       = 5     # draws per prompt variant
VOTE_THR      = 7     # majority threshold (7 out of 15 total draws)
SKILL_TOP_K   = 15    # number of skills to retrieve per document
FEWSHOT_K     = 4     # reduced demo count (skills provide knowledge, demos provide format)
RETRIEVAL_M   = 20    # BM25 top-M for demo sampling
MAX_TOKENS    = 16384
TEMPERATURE   = 0.0

PROMPTS = [
    ("v6", V6_SKILL_BASE, 20_000),
    ("v7", V7_SKILL_BASE, 30_000),
    ("v8", V8_SKILL_BASE, 40_000),
]


# ── Single draw ───────────────────────────────────────────────────────────────

def _one_draw(llm: LLM, text: str, system_prompt: str,
              demos: List[Dict], skills: List[str], seed: int) -> Dict[str, Any]:
    messages = build_messages_with_skills(system_prompt, text, demos, skills)
    outs, _ = llm.chat_with_thinking(messages, temperature=TEMPERATURE,
                                      n=1, seed=seed, max_tokens=MAX_TOKENS)
    raw = outs[0] if outs else ""
    parsed = extract_json_obj(raw) or {"entities": [], "relations": []}
    return to_official(text, parsed)


# ── Per-document extraction ───────────────────────────────────────────────────

def extract_doc(llm: LLM, text: str, bm25_retriever: BM25Retriever,
                skill_retriever: SkillRetriever, pool_docs: List[Dict],
                doc_idx: int) -> Dict[str, Any]:
    """Run all 15 draws (3 prompts × 5 draws) with skill retrieval."""
    skills = skill_retriever.retrieve(text, top_k=SKILL_TOP_K)

    all_draws = []
    for name, sys_prompt, seed_base in PROMPTS:
        top_m_idx = bm25_retriever.top_m(text, RETRIEVAL_M)
        for k in range(N_DRAWS):
            seed = seed_base + 97 * doc_idx + k
            demo_idx = random.Random(seed).sample(top_m_idx, min(FEWSHOT_K, len(top_m_idx)))
            demos = [pool_docs[i] for i in demo_idx]
            draw = _one_draw(llm, text, sys_prompt, demos, skills, seed)
            all_draws.append(draw)

    return majority_vote(all_draws, threshold=VOTE_THR)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    docs = json.load(open(args.input))
    print(f"[predict_skill] {len(docs)} input docs", flush=True)

    # Load demo pool
    pool_docs = json.load(open(args.pool))
    print(f"[predict_skill] pool={len(pool_docs)} train docs", flush=True)

    # Test mode: subset
    if args.test > 0:
        test_indices = random.Random(123).sample(range(len(docs)), min(args.test, len(docs)))
        docs = [docs[i] for i in test_indices]
        print(f"[predict_skill] TEST MODE: {len(docs)} docs", flush=True)

    # Build BM25 retriever for demo selection
    pool_texts = [d["text"] for d in pool_docs]
    print("[predict_skill] building BM25 retriever...", flush=True)
    bm25_retriever = BM25Retriever(pool_texts)
    print("[predict_skill] BM25 retriever ready", flush=True)

    # Build skill retriever
    print("[predict_skill] loading skill retriever...", flush=True)
    skill_ret = SkillRetriever(
        skills_json=args.skills,
        embeddings_npy=args.skill_embeddings,
    )
    print("[predict_skill] skill retriever ready", flush=True)

    # LLM
    llm = LLM(base_url=args.api_url)

    results: List[Optional[Dict]] = [None] * len(docs)
    t0 = time.time()
    n_done = 0

    def _worker(i_doc):
        i, doc = i_doc
        return i, extract_doc(llm, doc["text"], bm25_retriever, skill_ret, pool_docs, i)

    if args.test > 0:
        for i, doc in enumerate(docs):
            print(f"\n[predict_skill] Processing doc {i}: {doc['text'][:60]}...", flush=True)
            _, result = _worker((i, doc))
            results[i] = result
            print(f"  Predicted: {len(result.get('entities',[]))} entities, {len(result.get('relations',[]))} relations", flush=True)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, (i, d)): i for i, d in enumerate(docs)}
            with tqdm(total=len(docs), desc="[predict_skill]", unit="doc") as pbar:
                for fut in cf.as_completed(futures):
                    i = futures[fut]
                    try:
                        _, result = fut.result()
                        results[i] = result
                    except Exception as e:
                        tqdm.write(f"[doc {i}] failed: {e}")
                        results[i] = {"text": docs[i]["text"], "entities": [], "relations": []}
                    n_done += 1
                    dt = time.time() - t0
                    pbar.set_postfix(speed=f"{dt/n_done:.1f}s/doc")
                    pbar.update(1)

    preds = [r for r in results]
    _save(preds, args.output, args.zip)


def _save(preds, out_json, out_zip=""):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"[predict_skill] saved {len(preds)} docs -> {out_json}", flush=True)
    if out_zip:
        import zipfile
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_json, arcname="submit.json")
        print(f"[predict_skill] archive -> {out_zip}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Skill-augmented MGBIE predictor")
    p.add_argument("--api_url",  default=os.environ.get("MGBIE_BASE_URL", "http://localhost:8000/v1"),
                   help="LLM API base URL (or set MGBIE_BASE_URL env)")
    p.add_argument("--input",    required=True, help="Input JSON (test.json)")
    p.add_argument("--output",   required=True, help="Output prediction JSON")
    p.add_argument("--pool",     default="data/train.json", help="Demo pool JSON")
    p.add_argument("--skills",   default="data/skills_dedup.json", help="Deduplicated skills JSON")
    p.add_argument("--skill_embeddings", default="data/skills_embeddings.npy", help="Skill embeddings npy")
    p.add_argument("--workers",  type=int, default=32, help="Parallel workers")
    p.add_argument("--zip",      default="", help="Also write submit.zip")
    p.add_argument("--test",     type=int, default=0, help="Test mode: process N docs only")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
