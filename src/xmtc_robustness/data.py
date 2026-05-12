from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalDataset:
    """Dataset container used by the local experiment runner.

    The explicit text/matrix pairing keeps evaluation reproducible: the naive
    baseline only consumes `y_train`, while retrieval models consume `train_texts`
    plus the same labels, so both are tested on identical splits.
    """

    train_texts: list[str]
    test_texts: list[str]
    y_train: object
    y_test: object
    label_names: list[str] | None = None


def read_text_lines(path: str | Path) -> list[str]:
    """Read one UTF-8 document per line without stripping internal spaces."""

    file_path = Path(path)
    return file_path.read_text(encoding="utf-8").splitlines()


def load_local_dataset(dataset_dir: str | Path) -> LocalDataset:
    """Load a dataset folder containing texts and SciPy sparse label matrices."""

    from scipy import sparse

    root = Path(dataset_dir)
    required = ["train_texts.txt", "test_texts.txt", "y_train.npz", "y_test.npz"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Dataset folder is missing: {joined}")

    train_texts = read_text_lines(root / "train_texts.txt")
    test_texts = read_text_lines(root / "test_texts.txt")
    y_train = sparse.load_npz(root / "y_train.npz").tocsr()
    y_test = sparse.load_npz(root / "y_test.npz").tocsr()

    if len(train_texts) != y_train.shape[0]:
        raise ValueError("train_texts.txt line count must match y_train rows.")
    if len(test_texts) != y_test.shape[0]:
        raise ValueError("test_texts.txt line count must match y_test rows.")
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError("y_train and y_test must have the same number of labels.")

    labels_file = root / "labels.txt"
    label_names = read_text_lines(labels_file) if labels_file.exists() else None
    if label_names is not None and len(label_names) != y_train.shape[1]:
        raise ValueError("labels.txt line count must match number of labels.")

    return LocalDataset(
        train_texts=train_texts,
        test_texts=test_texts,
        y_train=y_train,
        y_test=y_test,
        label_names=label_names,
    )
