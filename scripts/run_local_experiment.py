from __future__ import annotations

import argparse
from pathlib import Path

from xmtc_robustness import (
    GlobalFrequencyRanker,
    InstanceKNNRetrievalRanker,
    compare_predictions,
    inverse_propensity_scores,
    label_counts,
)
from xmtc_robustness.data import load_local_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local XMTC robustness experiment.")
    parser.add_argument("--data", required=True, type=Path, help="Dataset folder under data/ or an absolute path.")
    parser.add_argument("--models", nargs="+", default=["baseline", "retrieval"], choices=["baseline", "retrieval"])
    parser.add_argument("--ks", nargs="+", default=[1, 3, 5], type=int, help="Values of k to evaluate.")
    parser.add_argument("--dataset-preset", default="default", help="Propensity preset, e.g. amazon-670k or wikilshtc.")
    parser.add_argument("--retrieval-backend", default="sklearn", choices=["auto", "faiss", "hnswlib", "sklearn"])
    parser.add_argument("--representation", default="tfidf", choices=["tfidf", "sentence-transformer"])
    parser.add_argument("--n-neighbors", default=32, type=int)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--skip-train-lines", default=0, type=int, help="Initial train text lines to skip, useful for headers.")
    parser.add_argument("--skip-test-lines", default=0, type=int, help="Initial test text lines to skip, useful for headers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_local_dataset(
        args.data,
        skip_train_lines=args.skip_train_lines,
        skip_test_lines=args.skip_test_lines,
    )

    predictions = {}
    max_k = max(args.ks)

    if "baseline" in args.models:
        baseline = GlobalFrequencyRanker().fit(dataset.y_train)
        predictions["global_frequency"] = baseline.predict_top_k(
            n_samples=dataset.y_test.shape[0],
            k=max_k,
        )

    if "retrieval" in args.models:
        retriever = InstanceKNNRetrievalRanker(
            representation=args.representation,
            backend=args.retrieval_backend,
            n_neighbors=args.n_neighbors,
            embedding_model=args.embedding_model,
        ).fit(dataset.train_texts, dataset.y_train)
        predictions["retrieval"] = retriever.predict_top_k(dataset.test_texts, k=max_k)

    inv_ps = inverse_propensity_scores.from_label_matrix(
        dataset.y_train,
        dataset=args.dataset_preset,
    )

    results = compare_predictions(
        dataset.y_test,
        predictions,
        ks=tuple(args.ks),
        inverse_propensity=inv_ps,
        counts=label_counts(dataset.y_train),
    )

    for model_name, metrics in results.items():
        print(f"\n[{model_name}]")
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name}: {value:.6f}")


if __name__ == "__main__":
    main()
