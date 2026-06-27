"""Core utilities for MGBIE extraction method.

Functions:
  extract_json_obj  — robust JSON extraction from LLM output
  to_official       — convert model output (occurrence-based) to official format (span-based)
  majority_vote     — aggregate N draws by vote threshold
  evaluate          — official NER/RE metrics
  format_metrics    — pretty-print metrics
"""
from __future__ import annotations
import json, re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from schema import ENT_LABELS, REL_LABELS


# ── JSON parsing ─────────────────────────────────────────────────────────────

def extract_json_obj(text: str) -> Optional[dict]:
    """Extract the last top-level JSON object; robust to <think> and markdown."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"```(?:json)?", "", text).replace("```", "")
    candidates, stack, start = [], [], None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack: start = i
            stack.append(i)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:i + 1]); start = None
    for cand in reversed(candidates):
        try: return json.loads(cand)
        except Exception: pass
    return None


# ── Entity alignment ──────────────────────────────────────────────────────────

def _find_all(text: str, needle: str) -> List[Tuple[int, int]]:
    out, start = [], 0
    while True:
        idx = text.find(needle, start)
        if idx < 0: break
        out.append((idx, idx + len(needle))); start = idx + 1
    return out


def align_entity(text: str, ent_text: str, occurrence: int = 1) -> Optional[Tuple[int, int]]:
    """Return (start, end) of the occurrence-th appearance of ent_text in text."""
    if not ent_text: return None
    occ = max(1, int(occurrence or 1))
    hits = _find_all(text, ent_text)
    if hits: return hits[min(occ, len(hits)) - 1]
    # case-insensitive fallback
    hits_ci = _find_all(text.lower(), ent_text.lower())
    if hits_ci:
        s, e = hits_ci[min(occ, len(hits_ci)) - 1]
        return s, e
    # whitespace-normalised
    m = list(re.finditer(re.sub(r"\\\s+", r"\\s+", re.escape(ent_text.strip())), text, re.IGNORECASE))
    if m: return m[min(occ - 1, len(m) - 1)].start(), m[min(occ - 1, len(m) - 1)].end()
    # strip punctuation
    stripped = ent_text.strip("\"'""''.,;:()[]{}")
    if stripped != ent_text: return align_entity(text, stripped, occurrence)
    return None


# ── Official format conversion ────────────────────────────────────────────────

def _demo_to_labeled_json(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert gold doc to the JSON shape the model must emit (for few-shot demos)."""
    text = doc["text"]
    seen: Dict[tuple, int] = {}
    ents_out = []
    for e in doc.get("entities", []):
        key = (e["text"], e["label"])
        seen[key] = seen.get(key, 0) + 1
        ents_out.append({"text": e["text"], "label": e["label"], "occurrence": seen[key]})

    def _occ_of(key: tuple, start: int) -> int:
        occ = 0
        for e in doc.get("entities", []):
            if (e["text"], e["label"]) == key:
                occ += 1
                if e["start"] == start: return occ
        return max(1, occ)

    rels_out = []
    for r in doc.get("relations", []):
        h_occ = _occ_of((r["head"], r["head_type"]), r["head_start"])
        t_occ = _occ_of((r["tail"], r["tail_type"]), r["tail_start"])
        rels_out.append({
            "head": r["head"], "head_label": r["head_type"], "head_occurrence": h_occ,
            "tail": r["tail"], "tail_label": r["tail_type"], "tail_occurrence": t_occ,
            "label": r["label"],
        })
    return {"entities": ents_out, "relations": rels_out}


def to_official(text: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert model output (occurrence-based) → official format (span-based)."""
    ents_out, span_map = [], {}
    for e in raw.get("entities", []) or []:
        if not isinstance(e, dict): continue
        etext, elabel = e.get("text") or "", e.get("label") or ""
        if elabel not in ENT_LABELS or not etext: continue
        occ = int(e.get("occurrence") or 1)
        span = align_entity(text, etext, occ)
        if span is None: continue
        s, t = span
        actual = text[s:t]
        ents_out.append({"start": s, "end": t, "text": actual, "label": elabel})
        span_map[(etext, elabel, occ)] = (s, t, actual)
        span_map.setdefault((etext, elabel, 1), (s, t, actual))

    seen_e = set(); ents_dedup = []
    for e in ents_out:
        k = (e["start"], e["end"], e["label"])
        if k not in seen_e: seen_e.add(k); ents_dedup.append(e)

    rels_out = []
    for r in raw.get("relations", []) or []:
        if not isinstance(r, dict): continue
        rlabel = r.get("label") or ""
        if rlabel not in REL_LABELS: continue
        h_text, t_text = r.get("head") or "", r.get("tail") or ""
        h_label, t_label = r.get("head_label") or "", r.get("tail_label") or ""
        h_occ, t_occ = int(r.get("head_occurrence") or 1), int(r.get("tail_occurrence") or 1)
        if h_label not in ENT_LABELS or t_label not in ENT_LABELS: continue
        hspan = span_map.get((h_text, h_label, h_occ)) or span_map.get((h_text, h_label, 1))
        tspan = span_map.get((t_text, t_label, t_occ)) or span_map.get((t_text, t_label, 1))
        if hspan is None:
            a = align_entity(text, h_text, h_occ)
            if a is None: continue
            hspan = (a[0], a[1], text[a[0]:a[1]])
        if tspan is None:
            a = align_entity(text, t_text, t_occ)
            if a is None: continue
            tspan = (a[0], a[1], text[a[0]:a[1]])
        rels_out.append({
            "head": hspan[2], "head_start": hspan[0], "head_end": hspan[1], "head_type": h_label,
            "tail": tspan[2], "tail_start": tspan[0], "tail_end": tspan[1], "tail_type": t_label,
            "label": rlabel,
        })

    seen_r = set(); rels_dedup = []
    for r in rels_out:
        k = (r["head_start"], r["head_end"], r["head_type"],
             r["tail_start"], r["tail_end"], r["tail_type"], r["label"])
        if k not in seen_r: seen_r.add(k); rels_dedup.append(r)

    return {"text": text, "entities": ents_dedup, "relations": rels_dedup}


# ── Majority vote ─────────────────────────────────────────────────────────────

def majority_vote(draws: List[Dict[str, Any]], threshold: int) -> Dict[str, Any]:
    """Keep entities/relations that appear in >= threshold draws."""
    assert draws
    text = draws[0]["text"]
    ent_cnt: Dict[tuple, int] = {}; ent_ex: Dict[tuple, Any] = {}
    rel_cnt: Dict[tuple, int] = {}; rel_ex: Dict[tuple, Any] = {}
    for d in draws:
        seen_e, seen_r = set(), set()
        for e in d.get("entities", []):
            k = (e["start"], e["end"], e["label"])
            if k not in seen_e:
                seen_e.add(k); ent_cnt[k] = ent_cnt.get(k, 0) + 1; ent_ex.setdefault(k, e)
        for r in d.get("relations", []):
            k = (r["head_start"], r["head_end"], r["head_type"],
                 r["tail_start"], r["tail_end"], r["tail_type"], r["label"])
            if k not in seen_r:
                seen_r.add(k); rel_cnt[k] = rel_cnt.get(k, 0) + 1; rel_ex.setdefault(k, r)
    return {
        "text": text,
        "entities":  [ent_ex[k] for k, c in ent_cnt.items() if c >= threshold],
        "relations": [rel_ex[k] for k, c in rel_cnt.items() if c >= threshold],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _ent_key(e): return (e["start"], e["end"], e["label"])
def _rel_key(r): return (r["head_start"], r["head_end"], r["head_type"],
                         r["tail_start"], r["tail_end"], r["tail_type"], r["label"])

def evaluate(preds: List[Dict], golds: List[Dict]) -> Dict:
    assert len(preds) == len(golds)
    ner_tp = ner_fp = ner_fn = re_tp = re_fp = re_fn = 0
    for p, g in zip(preds, golds):
        pe = {_ent_key(x) for x in p.get("entities", [])}
        ge = {_ent_key(x) for x in g.get("entities", [])}
        ner_tp += len(pe & ge); ner_fp += len(pe - ge); ner_fn += len(ge - pe)
        pr = {_rel_key(x) for x in p.get("relations", [])}
        gr = {_rel_key(x) for x in g.get("relations", [])}
        re_tp += len(pr & gr); re_fp += len(pr - gr); re_fn += len(gr - pr)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    np, nr, nf = prf(ner_tp, ner_fp, ner_fn)
    rp, rr, rf = prf(re_tp, re_fp, re_fn)
    ns = 0.5 * nf + 0.25 * np + 0.25 * nr
    rs = 0.5 * rf + 0.25 * rp + 0.25 * rr
    return {
        "ner": {"tp": ner_tp, "fp": ner_fp, "fn": ner_fn, "p": np, "r": nr, "f1": nf, "score": ns},
        "re":  {"tp": re_tp,  "fp": re_fp,  "fn": re_fn,  "p": rp, "r": rr, "f1": rf, "score": rs},
        "total": 0.4 * ns + 0.6 * rs,
    }


def format_metrics(m: Dict) -> str:
    n, r = m["ner"], m["re"]
    return (
        f"NER   P={n['p']:.4f} R={n['r']:.4f} F1={n['f1']:.4f} Score={n['score']:.4f}"
        f"  (tp={n['tp']} fp={n['fp']} fn={n['fn']})\n"
        f"RE    P={r['p']:.4f} R={r['r']:.4f} F1={r['f1']:.4f} Score={r['score']:.4f}"
        f"  (tp={r['tp']} fp={r['fp']} fn={r['fn']})\n"
        f"TOTAL = {m['total']:.4f}"
    )


# Public alias for external imports
demo_to_labeled_json = _demo_to_labeled_json
