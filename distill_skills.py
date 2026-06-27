"""Distill fine-grained reusable extraction skills from annotated training data.

Given each annotated training doc (text + entities + relations), call a strong LLM
to summarize transferable extraction rules/patterns that can help on unseen documents.

Usage:
  # Test on 5 random docs:
  python distill_skills.py --input data/train.json --output data/skills_raw.jsonl --test 5

  # Full distillation (all docs, 32 workers):
  python distill_skills.py --input data/train.json --output data/skills_raw.jsonl --workers 32
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, sys, time, random
from typing import Any, Dict, List, Optional
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import LLM
from schema import ENTITY_TYPES, RELATION_TYPES
from tools import extract_json_obj

# ── Distillation Prompt ───────────────────────────────────────────────────────

_ENT_GUIDE = "\n".join(f"  - {code} ({name}): {desc}" for code, name, desc in ENTITY_TYPES)
_REL_GUIDE = "\n".join(f"  - {code} ({name}, typical: {types}): {desc}" for code, name, types, desc in RELATION_TYPES)

DISTILL_SYSTEM_PROMPT = f"""You are an expert knowledge engineer for minor-grain crop breeding information extraction.

Your task: Given a passage with gold-standard entity and relation annotations, distill reusable, transferable "extraction skills" — fine-grained rules, patterns, or heuristics that would help correctly extract entities and relations from OTHER unseen passages.

Schema reference:
Entity types (12):
{_ENT_GUIDE}

Relation types (6):
{_REL_GUIDE}

Requirements for each skill:
1. Each skill must be ATOMIC — one specific, actionable rule/pattern/heuristic
2. Each skill must be TRANSFERABLE — useful for other passages, not specific to this one document
3. Each skill should be CONCRETE — include the pattern/signal to look for and what to extract
4. Categories: entity_recognition, relation_extraction, boundary_detection, disambiguation, co_reference

Output format — return a JSON object:
{{
  "skills": [
    {{
      "category": "<category>",
      "skill": "<concise rule/pattern description in English, 1-3 sentences>"
    }},
    ...
  ]
}}

Aim for 3-8 skills per document. Focus on non-obvious patterns that require domain knowledge or careful attention.
Do NOT output trivial skills like "extract all entities" or restate the schema definitions.
Do NOT mention specific entity text from this document unless it illustrates a general pattern."""


def _build_distill_user_msg(doc: Dict[str, Any]) -> str:
    """Build the user message showing the annotated document."""
    text = doc["text"]
    entities = doc.get("entities", [])
    relations = doc.get("relations", [])

    ent_lines = []
    for e in entities:
        ent_lines.append(f"  - \"{e['text']}\" → {e['label']} (span {e['start']}:{e['end']})")

    rel_lines = []
    for r in relations:
        rel_lines.append(
            f"  - {r['head']} ({r['head_type']}) --[{r['label']}]--> "
            f"{r['tail']} ({r['tail_type']})"
        )

    parts = [
        f"Passage:\n\"\"\"\n{text}\n\"\"\"",
        f"\nGold Entities ({len(entities)}):\n" + "\n".join(ent_lines) if ent_lines else "\nGold Entities: (none)",
        f"\nGold Relations ({len(relations)}):\n" + "\n".join(rel_lines) if rel_lines else "\nGold Relations: (none)",
        "\nDistill the reusable extraction skills from this annotated example. Return the JSON object."
    ]
    return "\n".join(parts)


# ── Single doc distillation ───────────────────────────────────────────────────

def distill_one(llm: LLM, doc: Dict[str, Any], doc_idx: int) -> Dict[str, Any]:
    """Distill skills from one annotated document."""
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": _build_distill_user_msg(doc)},
    ]
    outs, meta = llm.chat_with_thinking(
        messages, temperature=0.3, max_tokens=16384, n=1, seed=42 + doc_idx
    )
    raw_text = outs[0] if outs else ""
    parsed = extract_json_obj(raw_text)

    skills = []
    if parsed and "skills" in parsed:
        for s in parsed["skills"]:
            if isinstance(s, dict) and s.get("skill"):
                skills.append({
                    "category": s.get("category", "unknown"),
                    "skill": s["skill"].strip(),
                })

    return {
        "doc_idx": doc_idx,
        "n_skills": len(skills),
        "skills": skills,
        "raw_response": raw_text[:2000],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    docs = json.load(open(args.input))
    print(f"[distill] Loaded {len(docs)} docs from {args.input}", flush=True)

    # Test mode: random subset
    if args.test > 0:
        indices = random.Random(42).sample(range(len(docs)), min(args.test, len(docs)))
        print(f"[distill] TEST MODE: running on {len(indices)} random docs", flush=True)
    else:
        indices = list(range(len(docs)))

    # Resume: skip already-processed docs
    done_indices = set()
    if os.path.exists(args.output) and not args.test:
        with open(args.output) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_indices.add(obj["doc_idx"])
                except:
                    pass
        if done_indices:
            print(f"[distill] Resuming: {len(done_indices)} docs already done", flush=True)
            indices = [i for i in indices if i not in done_indices]

    if not indices:
        print("[distill] All docs already processed!", flush=True)
        return

    llm = LLM(base_url=args.api_url, request_timeout=300)
    print(f"[distill] Using LLM at {args.api_url}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    results = []
    t0 = time.time()

    def _worker(idx):
        return distill_one(llm, docs[idx], idx)

    if args.test > 0:
        for idx in indices:
            result = _worker(idx)
            results.append(result)
            print(f"\n{'='*60}")
            print(f"Doc {idx} (text preview: {docs[idx]['text'][:80]}...)")
            print(f"  Entities: {len(docs[idx].get('entities', []))}, Relations: {len(docs[idx].get('relations', []))}")
            print(f"  Distilled {result['n_skills']} skills:")
            for s in result["skills"]:
                print(f"    [{s['category']}] {s['skill']}")
            print(f"{'='*60}")

        out_path = args.output.replace(".jsonl", "_test.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[distill] Test results saved to {out_path}", flush=True)
        dt = time.time() - t0
        total_skills = sum(r["n_skills"] for r in results)
        print(f"[distill] {len(results)} docs, {total_skills} skills, {dt:.1f}s total", flush=True)
    else:
        with open(args.output, "a") as fout:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, idx): idx for idx in indices}
                with tqdm(total=len(indices), desc="[distill]", unit="doc") as pbar:
                    for fut in cf.as_completed(futures):
                        idx = futures[fut]
                        try:
                            result = fut.result()
                            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                            fout.flush()
                            results.append(result)
                        except Exception as e:
                            tqdm.write(f"[doc {idx}] FAILED: {e}")
                        pbar.update(1)

        dt = time.time() - t0
        total_skills = sum(r["n_skills"] for r in results)
        print(f"\n[distill] Done: {len(results)} docs, {total_skills} skills, {dt:.1f}s", flush=True)
        print(f"[distill] Output: {args.output}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Distill extraction skills from training data")
    p.add_argument("--api_url", default=os.environ.get("MGBIE_BASE_URL", "http://localhost:8000/v1"),
                   help="LLM API base URL (or set MGBIE_BASE_URL env)")
    p.add_argument("--input", default="data/train.json", help="Training data JSON")
    p.add_argument("--output", default="data/skills_raw.jsonl", help="Output JSONL")
    p.add_argument("--workers", type=int, default=32, help="Parallel workers")
    p.add_argument("--test", type=int, default=0, help="Test mode: distill N random docs only")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
