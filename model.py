"""Thin OpenAI-compatible chat client for LLM inference."""
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import requests


DEFAULT_BASE = os.environ.get("MGBIE_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("MGBIE_MODEL", "your-model-name")


class LLM:
    """OpenAI-compatible /chat/completions client with retry + raw-output capture.

    Supports models with `reasoning_content` (e.g. Qwen3-Thinking) where the
    `max_tokens` budget covers both reasoning and content tokens.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        model: str = DEFAULT_MODEL,
        api_key: str = "EMPTY",
        request_timeout: int = 900,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = request_timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 24576,
        n: int = 1,
        seed: Optional[int] = None,
        retries: int = 3,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        """Return list of n content-strings (thinking stripped)."""
        outs, _ = self.chat_with_thinking(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            retries=retries,
            enable_thinking=enable_thinking,
        )
        return outs

    def chat_with_thinking(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 24576,
        n: int = 1,
        seed: Optional[int] = None,
        retries: int = 3,
        enable_thinking: Optional[bool] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Return (contents, meta_list). meta_list[i] has finish_reason, reasoning_len, content_len."""
        url = f"{self.base_url}/chat/completions"

        def _payload(budget: int) -> Dict[str, Any]:
            p: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": budget,
                "n": n,
            }
            if seed is not None:
                p["seed"] = seed
            if enable_thinking is not None:
                p["chat_template_kwargs"] = {"thinking": bool(enable_thinking)}
            return p

        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_err = None
        budget = max_tokens
        for attempt in range(retries):
            try:
                r = self.session.post(url, json=_payload(budget), headers=headers, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {})
                outs: List[str] = []
                meta: List[Dict[str, Any]] = []
                needs_retry = False
                for ch in data["choices"]:
                    msg = ch.get("message", {})
                    content = (msg.get("content") or "").strip()
                    reasoning = msg.get("reasoning_content") or ""
                    fr = ch.get("finish_reason", "")
                    outs.append(content)
                    meta.append({
                        "finish_reason": fr,
                        "content_len": len(content),
                        "reasoning_len": len(reasoning),
                        "usage": usage,
                    })
                    # if thinking ate the budget leaving content empty, bump budget and retry
                    if fr == "length" and content == "" and attempt < retries - 1:
                        needs_retry = True
                if needs_retry:
                    budget = min(budget * 2, 65536)
                    print(f"[LLM] content empty (length-truncated); retrying with max_tokens={budget}")
                    continue
                return outs, meta
            except Exception as e:
                last_err = e
                wait = 2 * (attempt + 1)
                print(f"[LLM] attempt {attempt+1} failed: {e}; retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


if __name__ == "__main__":
    llm = LLM()
    outs, meta = llm.chat_with_thinking(
        [{"role": "user", "content": "Reply with the JSON {\"ok\": true}. No extra text."}],
        temperature=0.0,
        max_tokens=8192,
    )
    print("META:", meta)
    print("OUT:", outs[0])
