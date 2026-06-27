"""Three system prompts for the V6+V7+V8 ensemble.

V6 (strict typed-pair): emphasises typical head/tail type combinations.
V7 (no typed-pair):     only semantic definitions, any head/tail allowed.
V8 (balanced):          common patterns listed but explicitly not exhaustive.

All three share the same demo format and user template.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List
from schema import ENT_LABELS, REL_LABELS, ENTITY_TYPES
from tools import demo_to_labeled_json

_ENT_GUIDE = "\n".join(f"- {code} ({name}): {desc}" for code, name, desc in ENTITY_TYPES)

_USER_TMPL = 'Passage:\n"""\n{text}\n"""\n\nReturn the JSON object.'

_JSON_SCHEMA = (
    '{{\n'
    '  "entities": [{{"text": "<exact substring>", "label": "<' + '|'.join(ENT_LABELS) + '>", "occurrence": <int>}}],\n'
    '  "relations": [{{"head": "<text>", "head_label": "<lbl>", "head_occurrence": <int>,\n'
    '                 "tail": "<text>", "tail_label": "<lbl>", "tail_occurrence": <int>,\n'
    '                 "label": "<' + '|'.join(REL_LABELS) + '>"}}]\n'
    '}}'
)

# ── V6: strict typed-pair ─────────────────────────────────────────────────────
_V6_REL = "\n".join([
    "- CON (CONTAINS, typical: CROP→VAR): Variety belongs to a crop.",
    "- USE (USES, typical: VAR→BM): Breeding method used to produce a variety.",
    "- HAS (HAS, typical: VAR→TRT): Variety has / exhibits a trait.",
    "- AFF (AFFECTS, typical: ABS/GENE/MRK/QTL→TRT): Entity affects a trait.",
    "- OCI (OCCURS_IN, typical: TRT/ABS/BIS→GST): Trait/stress at a growth stage.",
    "- LOI (LOCATED_IN, typical: MRK/QTL/GENE→CHR): Entity located on a chromosome.",
])

V6_SYSTEM_PROMPT = f"""You are an information extraction system for minor-grain crop breeding texts.

Entities (12 types):
{_ENT_GUIDE}

Relations (6 types):
{_V6_REL}

Rules: entities are verbatim substrings; extract every mention (occurrence=1,2,3…); relations must use extracted entities as head/tail.

Return ONLY:
{_JSON_SCHEMA}"""

# ── V7: no typed-pair ─────────────────────────────────────────────────────────
_V7_REL = "\n".join([
    "- CON (CONTAINS): Head contains/subsumes the tail.",
    "- USE (USES): Head uses the tail as a method/marker/resource.",
    "- HAS (HAS): Head possesses/exhibits/is characterised by the tail trait.",
    "- AFF (AFFECTS): Head affects/regulates/correlates with the tail.",
    "- OCI (OCCURS_IN): Head occurs in/is measured at the tail growth stage.",
    "- LOI (LOCATED_IN/ASSOCIATED_WITH): Head is located in or associated with the tail.",
])

V7_SYSTEM_PROMPT = f"""You are an information extraction system for minor-grain crop breeding texts.

Entities (12 types):
{_ENT_GUIDE}

Relations (6 types — ANY head/tail type combination is valid when the semantic definition fits):
{_V7_REL}

Rules: entities are verbatim substrings; extract every mention (occurrence=1,2,3…); err on the side of recall for relations.

Return ONLY:
{_JSON_SCHEMA}"""

# ── V8: balanced ──────────────────────────────────────────────────────────────
_V8_REL = "\n".join([
    "- CON (CONTAINS, common: CROP→VAR): Head contains/subsumes the tail. Other combinations also valid.",
    "- USE (USES, common: VAR→BM): Head uses the tail as a method/resource. Other combinations also valid.",
    "- HAS (HAS, common: VAR→TRT): Head possesses/exhibits the tail. Other combinations also valid.",
    "- AFF (AFFECTS, common: ABS/GENE/MRK/QTL→TRT): Head affects/correlates with the tail. Other combinations also valid.",
    "- OCI (OCCURS_IN, common: TRT/ABS/BIS→GST): Head occurs at the tail stage. Other combinations also valid.",
    "- LOI (LOCATED_IN, common: MRK/QTL/GENE→CHR): Head located in/associated with the tail. Other combinations also valid.",
])

V8_SYSTEM_PROMPT = f"""You are an information extraction system for minor-grain crop breeding texts.

Entities (12 types):
{_ENT_GUIDE}

Relations (6 types — common patterns shown but NOT exhaustive; semantic definition is authoritative):
{_V8_REL}

Rules: entities are verbatim substrings; extract every mention (occurrence=1,2,3…); prefer common pattern when it applies but don't skip non-typical relations.

Return ONLY:
{_JSON_SCHEMA}"""

# ── Message builder ───────────────────────────────────────────────────────────

def build_messages(system_prompt: str, text: str, demos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build (system, user) messages with K few-shot demos followed by the query."""
    parts = []
    for i, d in enumerate(demos, 1):
        labeled = demo_to_labeled_json(d)
        parts.append(
            f"Example {i}:\nPassage:\n\"\"\"\n{d['text']}\n\"\"\"\nOutput:\n"
            + json.dumps(labeled, ensure_ascii=False)
        )
    parts.append("Now the real passage:\n" + _USER_TMPL.format(text=text))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]
