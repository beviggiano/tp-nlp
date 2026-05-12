from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np

from .global_frequency import GlobalFrequencyRanker


Representation = Literal["tfidf", "sentence-transformer", "sentence-transformers"]
Backend = Literal["auto", "faiss", "hnswlib", "sklearn"]


def _is_sparse_like(matrix: Any) -> bool:
    return hasattr(matrix, "tocsr") and not isinstance(matrix, np.ndarray)


def _as_text_list(texts: Iterable[str]) -> list[str]:
    if isinstance(texts, list):
        return texts
    return list(texts)


def _topk_dense(scores: np.ndarray, k: int, fallback: np.ndarray | None = None) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be 2D.")
    n_samples, n_labels = scores.shape
    if k > n_labels:
        raise ValueError(f"k={k} exceeds number of labels={n_labels}.")

    out = np.empty((n_samples, k), dtype=np.int64)
    for row in range(n_samples):
        row_scores = scores[row]
        positive = np.flatnonzero(row_scores > 0)
        if positive.size == 0 and fallback is not None:
            out[row] = fallback[:k]
            continue

        if positive.size > k:
            candidate = positive[np.argpartition(-row_scores[positive], k - 1)[:k]]
        else:
            candidate = positive
        order = np.lexsort((candidate, -row_scores[candidate]))
        chosen = candidate[order]
        if chosen.shape[0] < k and fallback is not None:
            chosen = _fill_with_fallback(chosen, fallback, k)
        out[row] = chosen[:k]
    return out


def _fill_with_fallback(chosen: np.ndarray, fallback: np.ndarray, k: int) -> np.ndarray:
    if chosen.shape[0] >= k:
        return chosen[:k]
    seen = set(chosen.tolist())
    fill = [label for label in fallback.tolist() if label not in seen]
    return np.asarray(chosen.tolist() + fill[: k - chosen.shape[0]], dtype=np.int64)


def _topk_sparse_rows(score_matrix: Any, k: int, fallback: np.ndarray) -> np.ndarray:
    scores = score_matrix.tocsr()
    n_samples, n_labels = scores.shape
    if k > n_labels:
        raise ValueError(f"k={k} exceeds number of labels={n_labels}.")

    out = np.empty((n_samples, k), dtype=np.int64)
    for row in range(n_samples):
        start, end = scores.indptr[row], scores.indptr[row + 1]
        labels = scores.indices[start:end]
        values = scores.data[start:end]

        nonzero = values > 0
        labels = labels[nonzero]
        values = values[nonzero]
        if labels.size == 0:
            out[row] = fallback[:k]
            continue

        if labels.size > k:
            candidate = np.argpartition(-values, k - 1)[:k]
            labels = labels[candidate]
            values = values[candidate]
        order = np.lexsort((labels, -values))
        chosen = labels[order]
        out[row] = _fill_with_fallback(chosen, fallback, k)
    return out


@dataclass
class InstanceKNNRetrievalRanker:
    """Text-aware retrieval model for scalable XMTC experiments.

    The model embeds training documents, retrieves nearest neighbors for each
    test document, and aggregates the neighbors' sparse labels. This creates a
    meaningful contrast with `GlobalFrequencyRanker`: wins should come from
    text similarity rather than global label popularity.
    """

    representation: Representation = "tfidf"
    backend: Backend = "auto"
    n_neighbors: int = 32
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    vectorizer_kwargs: dict[str, Any] | None = None
    dense_svd_components: int | None = None
    random_state: int = 13
    clip_negative_similarities: bool = True
    show_progress_bar: bool = False

    y_train_: Any | None = field(default=None, init=False)
    train_vectors_: Any | None = field(default=None, init=False)
    index_: Any | None = field(default=None, init=False)
    vectorizer_: Any | None = field(default=None, init=False)
    svd_: Any | None = field(default=None, init=False)
    encoder_: Any | None = field(default=None, init=False)
    fallback_ranking_: np.ndarray | None = field(default=None, init=False)
    backend_: str | None = field(default=None, init=False)
    n_train_: int | None = field(default=None, init=False)
    n_labels_: int | None = field(default=None, init=False)

    def fit(self, train_texts: Iterable[str], y_train: Any) -> "InstanceKNNRetrievalRanker":
        """Fit the text encoder, ANN index, and label aggregation fallback."""

        texts = _as_text_list(train_texts)
        if len(texts) != y_train.shape[0]:
            raise ValueError("train_texts length must match y_train rows.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")

        self.y_train_ = y_train.tocsr() if hasattr(y_train, "tocsr") else np.asarray(y_train)
        self.n_train_, self.n_labels_ = y_train.shape
        self.fallback_ranking_ = GlobalFrequencyRanker().fit(y_train).ranking_

        self.train_vectors_ = self._fit_transform_texts(texts)
        self.backend_ = self._resolve_backend(self.train_vectors_)
        self.index_ = self._build_index(self.train_vectors_, self.backend_)
        return self

    def predict_top_k(self, texts: Iterable[str], *, k: int = 5, n_neighbors: int | None = None) -> np.ndarray:
        """Predict labels by retrieving neighbors and aggregating their labels."""

        if self.index_ is None or self.y_train_ is None or self.fallback_ranking_ is None:
            raise RuntimeError("InstanceKNNRetrievalRanker must be fitted first.")
        if self.n_labels_ is None or k > self.n_labels_:
            raise ValueError(f"k={k} exceeds number of labels={self.n_labels_}.")

        query_texts = _as_text_list(texts)
        neighbors = min(n_neighbors or self.n_neighbors, self.n_train_ or self.n_neighbors)
        outputs: list[np.ndarray] = []

        for start in range(0, len(query_texts), self.batch_size):
            batch_texts = query_texts[start : start + self.batch_size]
            vectors = self._transform_texts(batch_texts)
            neighbor_ids, similarities = self._query_index(vectors, neighbors)
            outputs.append(self._neighbors_to_labels(neighbor_ids, similarities, k))

        return np.vstack(outputs) if outputs else np.empty((0, k), dtype=np.int64)

    def _fit_transform_texts(self, texts: list[str]) -> Any:
        if self.representation == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            kwargs = {"min_df": 1, "ngram_range": (1, 2), "dtype": np.float32}
            if self.vectorizer_kwargs:
                kwargs.update(self.vectorizer_kwargs)
            self.vectorizer_ = TfidfVectorizer(**kwargs)
            matrix = self.vectorizer_.fit_transform(texts)
            return self._maybe_project_dense(matrix)

        if self.representation in {"sentence-transformer", "sentence-transformers"}:
            from sentence_transformers import SentenceTransformer

            self.encoder_ = SentenceTransformer(self.embedding_model)
            vectors = self.encoder_.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=self.show_progress_bar,
            )
            return np.asarray(vectors, dtype=np.float32)

        raise ValueError(f"Unsupported representation: {self.representation}")

    def _transform_texts(self, texts: list[str]) -> Any:
        if self.representation == "tfidf":
            if self.vectorizer_ is None:
                raise RuntimeError("TF-IDF vectorizer is not fitted.")
            matrix = self.vectorizer_.transform(texts)
            return self._maybe_project_dense(matrix, fit=False)

        if self.encoder_ is None:
            raise RuntimeError("SentenceTransformer encoder is not fitted.")
        vectors = self.encoder_.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _maybe_project_dense(self, matrix: Any, *, fit: bool = True) -> Any:
        if self.dense_svd_components is None:
            return matrix

        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        if fit:
            self.svd_ = TruncatedSVD(
                n_components=self.dense_svd_components,
                random_state=self.random_state,
            )
            dense = self.svd_.fit_transform(matrix)
        else:
            if self.svd_ is None:
                raise RuntimeError("SVD projector is not fitted.")
            dense = self.svd_.transform(matrix)
        return normalize(dense, norm="l2").astype(np.float32, copy=False)

    def _resolve_backend(self, vectors: Any) -> str:
        if self.backend != "auto":
            if self.backend in {"faiss", "hnswlib"} and _is_sparse_like(vectors):
                raise ValueError(
                    f"{self.backend} requires dense vectors. Use Sentence-Transformers "
                    "or set dense_svd_components for TF-IDF."
                )
            return self.backend

        if not _is_sparse_like(vectors):
            try:
                import faiss  # noqa: F401

                return "faiss"
            except ImportError:
                pass
            try:
                import hnswlib  # noqa: F401

                return "hnswlib"
            except ImportError:
                pass
        return "sklearn"

    def _build_index(self, vectors: Any, backend: str) -> Any:
        if backend == "faiss":
            import faiss

            dense = np.asarray(vectors, dtype=np.float32)
            index = faiss.IndexFlatIP(dense.shape[1])
            index.add(dense)
            return index

        if backend == "hnswlib":
            import hnswlib

            dense = np.asarray(vectors, dtype=np.float32)
            index = hnswlib.Index(space="cosine", dim=dense.shape[1])
            index.init_index(max_elements=dense.shape[0], ef_construction=200, M=32)
            index.add_items(dense, np.arange(dense.shape[0]))
            index.set_ef(max(50, self.n_neighbors * 2))
            return index

        if backend == "sklearn":
            from sklearn.neighbors import NearestNeighbors

            index = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=self.n_neighbors)
            index.fit(vectors)
            return index

        raise ValueError(f"Unsupported backend: {backend}")

    def _query_index(self, vectors: Any, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        if self.backend_ == "faiss":
            sims, ids = self.index_.search(np.asarray(vectors, dtype=np.float32), n_neighbors)
        elif self.backend_ == "hnswlib":
            ids, distances = self.index_.knn_query(np.asarray(vectors, dtype=np.float32), k=n_neighbors)
            sims = 1.0 - distances
        elif self.backend_ == "sklearn":
            distances, ids = self.index_.kneighbors(vectors, n_neighbors=n_neighbors, return_distance=True)
            sims = 1.0 - distances
        else:
            raise RuntimeError("Unknown backend state.")

        sims = np.asarray(sims, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if self.clip_negative_similarities:
            sims = np.maximum(sims, 0.0)
        return ids, sims

    def _neighbors_to_labels(self, neighbor_ids: np.ndarray, similarities: np.ndarray, k: int) -> np.ndarray:
        if _is_sparse_like(self.y_train_):
            from scipy import sparse

            n_queries, neighbors = neighbor_ids.shape
            row_ids = np.repeat(np.arange(n_queries), neighbors)
            weights = sparse.csr_matrix(
                (similarities.reshape(-1), (row_ids, neighbor_ids.reshape(-1))),
                shape=(n_queries, self.n_train_),
            )
            label_scores = weights @ self.y_train_
            return _topk_sparse_rows(label_scores, k, self.fallback_ranking_)

        y_dense = np.asarray(self.y_train_, dtype=np.float32)
        scores = np.zeros((neighbor_ids.shape[0], y_dense.shape[1]), dtype=np.float32)
        for row, (ids, sims) in enumerate(zip(neighbor_ids, similarities, strict=True)):
            scores[row] = sims @ y_dense[ids]
        return _topk_dense(scores, k, self.fallback_ranking_)
