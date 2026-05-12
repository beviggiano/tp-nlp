from __future__ import annotations

from scipy import sparse

from xmtc_robustness import (
    GlobalFrequencyRanker,
    InstanceKNNRetrievalRanker,
    compare_predictions,
    inverse_propensity_scores,
    label_counts,
)


train_texts = [
    "neural networks for language models",
    "graph algorithms and shortest paths",
    "deep learning for text classification",
    "rare medical entity extraction",
]
test_texts = [
    "language classification with neural models",
    "medical extraction task",
]

y_train = sparse.csr_matrix(
    [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ],
    dtype="float32",
)
y_test = sparse.csr_matrix(
    [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ],
    dtype="float32",
)

baseline = GlobalFrequencyRanker().fit(y_train)
baseline_pred = baseline.predict_top_k(n_samples=y_test.shape[0], k=3)

retriever = InstanceKNNRetrievalRanker(
    representation="tfidf",
    backend="sklearn",
    n_neighbors=2,
).fit(train_texts, y_train)
retriever_pred = retriever.predict_top_k(test_texts, k=3)

inv_ps = inverse_propensity_scores.from_label_matrix(y_train, dataset="default")
results = compare_predictions(
    y_test,
    {"global_frequency": baseline_pred, "retrieval": retriever_pred},
    ks=(1, 3),
    inverse_propensity=inv_ps,
    counts=label_counts(y_train),
)

for model_name, metrics in results.items():
    print(model_name)
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
