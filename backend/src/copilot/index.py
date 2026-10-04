"""
Retrieval indexes. All three are scikit-learn/scipy only, so the API fits in a
serverless function without torch or a downloaded embedding model.

* ``BM25Index``: Okapi BM25 lexical ranking. The API default, because it
  scored best on the held-out SciFact benchmark (see ``eval_scifact.py``).
* ``TfidfIndex``: cosine similarity over TF-IDF vectors.
* ``LsaIndex``: TF-IDF followed by truncated SVD (latent semantic analysis).

Each index ranks chunks; the API and the benchmark both go through the same
``fit`` / ``query`` interface.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .schema import Chunk, RetrievedChunk


class _BaseIndex:
    name = "base"

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]) -> _BaseIndex:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        self.chunks = list(chunks)
        self._fit([c.text for c in self.chunks])
        return self

    def scores(self, text: str) -> np.ndarray:
        if not self.chunks:
            raise RuntimeError("Index has not been fit yet")
        return self._scores(text)

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        sims = self.scores(text)
        top_k = max(1, min(top_k, len(sims)))
        top = np.argpartition(-sims, top_k - 1)[:top_k]
        ranked = top[np.argsort(-sims[top], kind="stable")]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(sims[i])) for i in ranked]

    @property
    def size(self) -> int:
        return len(self.chunks)

    def _fit(self, texts: list[str]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _scores(self, text: str) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class BM25Index(_BaseIndex):
    """Okapi BM25 with the usual k1 (term-frequency saturation) and b (length normalisation)."""

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        super().__init__()
        self.k1, self.b = k1, b

    def _fit(self, texts: list[str]) -> None:
        self.vectorizer = CountVectorizer(stop_words="english")
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean() if doc_len.mean() > 0 else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

        # Precompute the BM25 weight of every (doc, term) pair so a query is one sparse product.
        norm = self.k1 * (1 - self.b + self.b * doc_len / avg_len)
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        weights = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * self.idf[tf.indices]
        self.matrix = sparse.csr_matrix((weights, tf.indices, tf.indptr), shape=tf.shape)

    def _scores(self, text: str) -> np.ndarray:
        q = self.vectorizer.transform([text])
        q.data[:] = 1.0  # each distinct query term counts once
        return np.asarray((self.matrix @ q.T).todense()).ravel()


class TfidfIndex(_BaseIndex):
    name = "tfidf"

    def __init__(self, sublinear_tf: bool = True) -> None:
        super().__init__()
        self.sublinear_tf = sublinear_tf

    def _fit(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=self.sublinear_tf)
        self.matrix = self.vectorizer.fit_transform(texts)

    def _scores(self, text: str) -> np.ndarray:
        return np.asarray((self.matrix @ self.vectorizer.transform([text]).T).todense()).ravel()


class LsaIndex(_BaseIndex):
    name = "lsa"

    def __init__(self, n_components: int = 150, max_features: int = 50_000, random_state: int = 7) -> None:
        super().__init__()
        self.n_components, self.max_features, self.random_state = n_components, max_features, random_state

    def _fit(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=self.max_features)
        tfidf = self.vectorizer.fit_transform(texts)
        # n_components must stay below both the number of chunks and the vocabulary size.
        n_components = max(1, min(self.n_components, len(texts) - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.vectors = normalize(self.svd.fit_transform(tfidf))

    def _scores(self, text: str) -> np.ndarray:
        q = normalize(self.svd.transform(self.vectorizer.transform([text])))
        return (self.vectors @ q.T).ravel()


INDEXES = {"bm25": BM25Index, "tfidf": TfidfIndex, "lsa": LsaIndex}
DEFAULT_INDEX = "bm25"
# Tuned on the SciFact train split (see eval_scifact.py and benchmark.json).
DEFAULT_PARAMS = {"k1": 0.9, "b": 0.5}


def make_index(name: str = DEFAULT_INDEX, **params) -> _BaseIndex:
    if name == DEFAULT_INDEX and not params:
        params = dict(DEFAULT_PARAMS)
    try:
        return INDEXES[name](**params)
    except KeyError:
        raise ValueError(f"Unknown index '{name}'. Choose from: {', '.join(INDEXES)}") from None


# Backwards-compatible name for the original LSA index.
RetrievalIndex = LsaIndex
