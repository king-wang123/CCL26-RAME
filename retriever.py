"""BM25 retriever over character n-grams for demo selection."""
from __future__ import annotations
import math, random, re
from collections import Counter, defaultdict
from typing import List

_SPACE = re.compile(r"\s+")

def _ngrams(text: str, nmin: int = 3, nmax: int = 6) -> List[str]:
    t = _SPACE.sub(" ", text.lower()).strip()
    out = []
    for n in range(nmin, nmax + 1):
        for i in range(len(t) - n + 1):
            out.append(t[i:i + n])
    return out


class BM25Retriever:
    """BM25 over char-ngram tokens (k1=1.2, b=0.75)."""

    def __init__(self, docs: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        tokenised = [_ngrams(d) for d in docs]
        self.doc_len = [len(x) for x in tokenised]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf = [dict(Counter(x)) for x in tokenised]
        df: dict = defaultdict(int)
        for toks in tokenised:
            for w in set(toks): df[w] += 1
        N = len(docs)
        self.idf = {w: math.log(1 + (N - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def top_m(self, query: str, m: int) -> List[int]:
        q = _ngrams(query); qf = Counter(q)
        scores = [0.0] * len(self.tf)
        for w, _ in qf.items():
            idf = self.idf.get(w, 0.0)
            if not idf: continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f: continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
                scores[i] += idf * (f * (self.k1 + 1)) / max(1e-9, denom)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:m]

    def sample_k(self, query: str, m: int, k: int, seed: int) -> List[int]:
        """Retrieve top-m by BM25, then random.sample k for draw diversity."""
        top = self.top_m(query, m)
        k = min(k, len(top))
        return random.Random(seed).sample(top, k)
