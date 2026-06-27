"""MGBIE predictor: V6+V7+V8 three-prompt ensemble with retrieval-augmented generation.

Method:
  Three prompts (V6 strict-typed-pair / V7 no-typed-pair / V8 balanced),
  each running N_DRAWS independent draws with retrieved K=8 demos from a pool
  of training docs. Total 3*N_DRAWS draws per document, majority vote at
  configurable threshold for both entities and relations.

Usage:
  # Inference on test set (BM25-only retrieval):
  python predict.py --input data/test.json --output results/submit.json \
      --pool data/train.json --workers 16 --retriever bm25

  # Inference with hybrid retrieval (BM25 + Embedding RRF):
  python predict.py --input data/test.json --output results/submit.json \
      --pool data/train.json --workers 16 --retriever hybrid

  # Evaluate on dev split:
  python predict.py --input data/train.json --dev_split data/dev_split.json \
      --output results/dev_pred.json --workers 16

  # Merge pre-computed raw files (skip LLM calls):
  python predict.py --merge_v6 v6.raw.json --merge_v7 v7.raw.json \
      --merge_v8 v8.raw.json --input data/test.json --output results/submit.json
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, sys, time
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import LLM
from retriever import BM25Retriever
from prompts import V6_SYSTEM_PROMPT, V7_SYSTEM_PROMPT, V8_SYSTEM_PROMPT, build_messages
from tools import extract_json_obj, to_official, majority_vote

# ── Hyperparameters (frozen configuration) ─────────────────────────────────────
N_DRAWS    = 60    # draws per prompt variant
VOTE_THR   = 91    # majority threshold (out of 3*N_DRAWS total draws)
FEWSHOT_K  = 8     # demos per draw
RETRIEVAL_M = 30   # BM25 top-M to sample K from
MAX_TOKENS = 65536 # token budget (covers reasoning + content)
TEMPERATURE = 0.0
ENABLE_THINKING = True  # enable thinking/reasoning mode for quality

PROMPTS = [
    ("v6", V6_SYSTEM_PROMPT, 20_000),  # seed_base offsets to avoid collision
    ("v7", V7_SYSTEM_PROMPT, 30_000),
    ("v8", V8_SYSTEM_PROMPT, 40_000),
]


# ── Single draw ───────────────────────────────────────────────────────────────

def _one_draw(llm: LLM, text: str, system_prompt: str,
              demos: List[Dict], seed: int) -> Dict[str, Any]:
    messages = build_messages(system_prompt, text, demos)
    outs, _ = llm.chat_with_thinking(messages, temperature=TEMPERATURE,
                                      n=1, seed=seed, max_tokens=MAX_TOKENS,
                                      enable_thinking=ENABLE_THINKING)
    raw = outs[0] if outs else ""
    parsed = extract_json_obj(raw) or {"entities": [], "relations": []}
    return to_official(text, parsed)


# ── Per-document extraction ───────────────────────────────────────────────────

def extract_doc(llm: LLM, text: str, retriever,
                pool_docs: List[Dict], doc_idx: int) -> Dict[str, Any]:
    """Run all draws (3 prompts × N_DRAWS) and return majority-voted result."""
    all_draws = []
    for name, sys_prompt, seed_base in PROMPTS:
        top_m_idx = retriever.top_m(text, RETRIEVAL_M)
        for k in range(N_DRAWS):
            seed = seed_base + 97 * doc_idx + k
            demo_local_idx = random.Random(seed).sample(top_m_idx, min(FEWSHOT_K, len(top_m_idx)))
            demos = [pool_docs[i] for i in demo_local_idx]
            draw = _one_draw(llm, text, sys_prompt, demos, seed)
            all_draws.append(draw)
    return majority_vote(all_draws, threshold=VOTE_THR)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    # Load input docs
    docs = json.load(open(args.input))
    print(f"[predict] {len(docs)} input docs", flush=True)

    # Merge mode: skip LLM calls, just merge pre-computed raw files
    if args.merge_v6 and args.merge_v7 and args.merge_v8:
        raw6 = json.load(open(args.merge_v6))
        raw7 = json.load(open(args.merge_v7))
        raw8 = json.load(open(args.merge_v8))
        assert len(raw6) == len(raw7) == len(raw8) == len(docs)
        preds = []
        for r6, r7, r8, doc in zip(raw6, raw7, raw8, docs):
            draws = (r6.get("draws", [])[:N_DRAWS]
                     + r7.get("draws", [])[:N_DRAWS]
                     + r8.get("draws", [])[:N_DRAWS])
            if not draws:
                preds.append({"text": doc["text"], "entities": [], "relations": []})
            else:
                preds.append(majority_vote(draws, threshold=VOTE_THR))
        _save(preds, args.output, args.zip)
        return

    # Load demo pool
    if args.dev_split:
        split = json.load(open(args.dev_split))
        all_train = json.load(open(args.pool or args.input))
        pool_docs = [all_train[i] for i in split["pool_idx"]]
        # Evaluate mode: only run on dev docs
        dev_idx = split["dev_idx"]
        docs = [all_train[i] for i in dev_idx]
        print(f"[predict] dev mode: {len(docs)} dev docs, pool={len(pool_docs)}", flush=True)
    else:
        pool_docs = json.load(open(args.pool))
        print(f"[predict] inference mode: pool={len(pool_docs)} train docs", flush=True)

    # Build retriever
    pool_texts = [d["text"] for d in pool_docs]
    print("[predict] building BM25 retriever...", flush=True)
    bm25_retriever = BM25Retriever(pool_texts)
    print("[predict] BM25 retriever ready", flush=True)

    if args.retriever == "hybrid":
        from embedding_retriever import EmbeddingRetriever, HybridRetriever
        print("[predict] building Embedding retriever...", flush=True)
        cache_path = args.embedding_cache or None
        emb_retriever = EmbeddingRetriever(pool_texts, batch_size=64, cache_path=cache_path)
        retriever = HybridRetriever(bm25_retriever, emb_retriever, rrf_k=60)
        print("[predict] Hybrid retriever ready", flush=True)
    else:
        retriever = bm25_retriever
        print("[predict] using BM25-only retriever", flush=True)

    llms = [LLM(base_url=args.api_url, model=args.model)]

    # ── Checkpoint: incremental save ──────────────────────────────────────────
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), ".checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(idx: int) -> str:
        return os.path.join(checkpoint_dir, f"doc_{idx:06d}.json")

    def _save_checkpoint(idx: int, result: Dict):
        path = _get_checkpoint_path(idx)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f, ensure_ascii=False)
        os.replace(tmp_path, path)  # atomic on POSIX

    def _load_checkpoint(idx: int) -> Optional[Dict]:
        path = _get_checkpoint_path(idx)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    # Determine which docs are already done
    done_indices = set()
    results: List[Optional[Dict]] = [None] * len(docs)
    for i in range(len(docs)):
        cached = _load_checkpoint(i)
        if cached is not None:
            results[i] = cached
            done_indices.add(i)

    remaining = [(i, docs[i]) for i in range(len(docs)) if i not in done_indices]
    print(f"[predict] checkpoint: {len(done_indices)} done, {len(remaining)} remaining", flush=True)

    if not remaining:
        print("[predict] all docs already completed from checkpoint!", flush=True)
    else:
        t0 = time.time()
        n_done = 0

        def _worker(i_doc):
            i, doc = i_doc
            return i, extract_doc(random.choice(llms), doc["text"], retriever, pool_docs, i)

        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, (i, d)): i for i, d in enumerate(docs) if i not in done_indices}
            with tqdm(total=len(remaining), desc="[predict]", unit="doc",
                      initial=0) as pbar:
                for fut in cf.as_completed(futures):
                    i = futures[fut]
                    try:
                        _, result = fut.result()
                        results[i] = result
                        _save_checkpoint(i, result)
                    except Exception as e:
                        tqdm.write(f"[doc {i}] failed: {e}")
                        fallback = {"text": docs[i]["text"], "entities": [], "relations": []}
                        results[i] = fallback
                        _save_checkpoint(i, fallback)
                    n_done += 1
                    dt = time.time() - t0
                    pbar.set_postfix(speed=f"{dt/n_done:.1f}s/doc")
                    pbar.update(1)

    # Assemble final results in original order
    preds = [results[i] for i in range(len(docs))]
    _save(preds, args.output, args.zip)
    print(f"[predict] checkpoint dir kept at: {checkpoint_dir}", flush=True)

    # Evaluate if gold available
    if args.dev_split:
        from tools import evaluate, format_metrics
        print(format_metrics(evaluate(preds, docs)), flush=True)


def _save(preds, out_json, out_zip=""):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"[predict] saved {len(preds)} docs -> {out_json}", flush=True)
    if out_zip:
        import zipfile
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_json, arcname="submit.json")
        print(f"[predict] archive -> {out_zip}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="MGBIE predictor (V6+V7+V8 ensemble)")
    p.add_argument("--api_url",  default=os.environ.get("MGBIE_BASE_URL", "http://localhost:8000/v1"),
                   help="OpenAI-compatible API base URL (or set MGBIE_BASE_URL env)")
    p.add_argument("--model",    default=os.environ.get("MGBIE_MODEL", "your-model-name"),
                   help="Model name for API calls (or set MGBIE_MODEL env)")
    p.add_argument("--input",    required=True, help="Input JSON (test.json or train.json)")
    p.add_argument("--output",   required=True, help="Output prediction JSON")
    p.add_argument("--pool",     default="data/train.json", help="Demo pool JSON (default: data/train.json)")
    p.add_argument("--dev_split",default="", help="dev_split.json for dev evaluation mode")
    p.add_argument("--workers",  type=int, default=32, help="Parallel workers")
    p.add_argument("--zip",      default="", help="Also write submit.zip")
    p.add_argument("--retriever", default="hybrid", choices=["hybrid", "bm25"],
                   help="Retrieval method: 'hybrid' (BM25+Embedding RRF) or 'bm25' (BM25 only)")
    p.add_argument("--embedding_cache", default="data/train_embeddings.npy",
                   help="Path to pre-computed embedding .npy cache")
    p.add_argument("--merge_v6", default="", help="Pre-computed V6 raw JSON (skip LLM calls)")
    p.add_argument("--merge_v7", default="", help="Pre-computed V7 raw JSON")
    p.add_argument("--merge_v8", default="", help="Pre-computed V8 raw JSON")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
