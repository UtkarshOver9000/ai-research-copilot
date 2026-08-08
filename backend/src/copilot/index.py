"""
Lightweight semantic-ish retrieval index: TF-IDF + truncated SVD (classic
Latent Semantic Analysis), not a transformer embedding model.

This is a deliberate tradeoff: a dense neural embedding model (e.g.
sentence-transformers) gives better retrieval quality but pulls in torch
and a multi-hundred-MB model, which doesn't fit in a serverless deployment
with a 50MB lambda limit. TF-IDF+SVD is scikit-learn only, trains in
milliseconds, and still captures co-occurrence-based topical similarity
well enough to be genuinely useful -- see README for measured Recall@k/MRR
rather than a marketing claim either way.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import Chunk, RetrievedChunk


class RetrievalIndex:
    def __init__(self, max_features: int = 5000, n_components: int = 150, random_state: int = 7):
        self.max_features = max_features
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.chunks: list[Chunk] = []
        self._chunk_vectors: np.ndarray | None = None

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english")
        tfidf = self.vectorizer.fit_transform(texts)

        # n_components must stay below both the sample count and the actual
        # vocabulary size TruncatedSVD ends up with, not just the configured cap.
        n_components = min(self.n_components, max(2, len(chunks) - 1), tfidf.shape[1] - 1)

        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self._chunk_vectors = self.svd.fit_transform(tfidf)

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.vectorizer is None or self.svd is None or self._chunk_vectors is None:
            raise RuntimeError("Index has not been fit yet")

        query_tfidf = self.vectorizer.transform([text])
        query_vector = self.svd.transform(query_tfidf)

        similarities = cosine_similarity(query_vector, self._chunk_vectors)[0]
        ranked_indices = np.argsort(-similarities)[:top_k]

        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(similarities[i]))
            for i in ranked_indices
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)
