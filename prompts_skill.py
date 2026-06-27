"""Skill-augmented prompt templates for V6+V7+V8 ensemble.

Injects retrieved extraction skills as "Extraction Guidelines" into the prompt,
while keeping a smaller number of few-shot demos for output format reference.
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

# ── Skill-augmented system prompts ─────────────────────────────────────────────

_SKILL_SECTION_HEADER = """
Extraction Guidelines (retrieved domain knowledge — apply these rules when relevant):
"""

# V6: strict typed-pair + skills
_V6_REL = "\n".join([
    "- CON (CONTAINS, typical: CROP→VAR): Variety belongs to a crop.",
    "- USE (USES, typical: VAR→BM): Breeding method used to produce a variety.",
    "- HAS (HAS, typical: VAR→TRT): Variety has / exhibits a trait.",
    "- AFF (AFFECTS, typical: ABS/GENE/MRK/QTL→TRT): Entity affects a trait.",
    "- OCI (OCCURS_IN, typical: TRT/ABS/BIS→GST): Trait/stress at a growth stage.",
    "- LOI (LOCATED_IN, typical: MRK/QTL/GENE→CHR): Entity located on a chromosome.",
])

_V6_BASE = f"""You are an information extraction system for minor-grain crop breeding texts.

Entities (12 types):
{_ENT_GUIDE}

Relations (6 types):
{_V6_REL}

Rules: entities are verbatim substrings; extract every mention (occurrence=1,2,3…); relations must use extracted entities as head/tail.

Return ONLY:
{_JSON_SCHEMA}"""

# V7: no typed-pair + skills
_V7_REL = "\n".join([
    "- CON (CONTAINS): Head contains/subsumes the tail.",
    "- USE (USES): Head uses the tail as a method/marker/resource.",
    "- HAS (HAS): Head possesses/exhibits/is characterised by the tail trait.",
    "- AFF (AFFECTS): Head affects/regulates/correlates with the tail.",
    "- OCI (OCCURS_IN): Head occurs in/is measured at the tail growth stage.",
    "- LOI (LOCATED_IN/ASSOCIATED_WITH): Head is located in or associated with the tail.",
])

_V7_BASE = f"""You are an information extraction system for minor-grain crop breeding texts.

Entities (12 types):
{_ENT_GUIDE}

Relations (6 types — ANY head/tail type combination is valid when the semantic definition fits):
{_V7_REL}

Rules: entities are verbatim substrings; extract every mention (occurrence=1,2,3…); err on the side of recall for relations.

Return ONLY:
{_JSON_SCHEMA}"""

# V8: balanced + skills
_V8_REL = "\n".join([
    "- CON (CONTAINS, common: CROP→VAR): Head contains/subsumes the tail. Other combinations also valid.",
    "- USE (USES, common: VAR→BM): Head uses the tail as a method/resource. Other combinations also valid.",
    "- HAS (HAS, common: VAR→TRT): Head possesses/exhibits the tail. Other combinations also valid.",
    "- AFF (AFFECTS, common: ABS/GENE/MRK/QTL→TRT): Head affects/correlates with the tail. Other combinations also valid.",
    "- OCI (OCCURS_IN, common: TRT/ABS/BIS→GST): Head occurs at the tail stage. Other combinations also valid.",
    "- LOI (LOCATED_IN, common: MRK/QTL/GENE→CHR): Head located in/associated with the tail. Other combinations also valid.",
])

_V8_BASE = f"""You are an information extraction system for minor-grain crop breeding texts.

Entities (12 types):
{_ENT_GUIDE}

Relations (6 types — common patterns shown but NOT exhaustive; semantic definition is authoritative):
{_V8_REL}

Rules: entities are verbatim substrings; extract every mention (occurrence=1,2,3…); prefer common pattern when it applies but don't skip non-typical relations.

Return ONLY:
{_JSON_SCHEMA}"""


def _build_system_with_skills(base_prompt: str, skills: List[str]) -> str:
    """Inject retrieved skills into the system prompt."""
    if not skills:
        return base_prompt
    skill_block = _SKILL_SECTION_HEADER + "\n".join(f"{i+1}. {s}" for i, s in enumerate(skills))
    return base_prompt + "\n" + skill_block


def get_skill_system_prompts(skills: List[str]):
    """Return (v6, v7, v8) system prompts with skills injected."""
    return (
        _build_system_with_skills(_V6_BASE, skills),
        _build_system_with_skills(_V7_BASE, skills),
        _build_system_with_skills(_V8_BASE, skills),
    )


# ── Message builder (skill version) ──────────────────────────────────────────

def build_messages_with_skills(
    system_prompt: str,
    text: str,
    demos: List[Dict[str, Any]],
    skills: List[str],
) -> List[Dict[str, str]]:
    """Build messages with skills in system prompt + few-shot demos for format reference."""
    full_system = _build_system_with_skills(system_prompt, skills)

    parts = []
    for i, d in enumerate(demos, 1):
        labeled = demo_to_labeled_json(d)
        parts.append(
            f"Example {i}:\nPassage:\n\"\"\"\n{d['text']}\n\"\"\"\nOutput:\n"
            + json.dumps(labeled, ensure_ascii=False)
        )
    parts.append("Now the real passage:\n" + _USER_TMPL.format(text=text))

    return [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]


# Pre-built base prompts (without skills) for PROMPTS list in predict_skill.py
V6_SKILL_BASE = _V6_BASE
V7_SKILL_BASE = _V7_BASE
V8_SKILL_BASE = _V8_BASE
