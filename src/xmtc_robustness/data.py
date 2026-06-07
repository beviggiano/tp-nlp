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


@dataclass(frozen=True)
class LocalDatasetSummary:
    """Lightweight shape/count summary used to diagnose dataset folders."""

    dataset_dir: Path
    train_text_lines: int | None
    test_text_lines: int | None
    y_train_shape: tuple[int, int] | None
    y_test_shape: tuple[int, int] | None
    y_train_nnz: int | None
    y_test_nnz: int | None
    label_lines: int | None
    missing_files: tuple[str, ...]


def count_text_lines(path: str | Path) -> int:
    """Count logical text records using the same delimiter as the loader."""

    return len(read_text_lines(path))


def read_text_lines(path: str | Path, *, skip_lines: int = 0) -> list[str]:
    """Read one UTF-8 document per `\\n` record.

    We intentionally split only on `\\n`, not on every Unicode line separator.
    Some large XMTC files contain stray carriage-return characters inside a
    record; Python's `splitlines()` would treat them as new documents and break
    the row alignment with the sparse matrix.
    """

    if skip_lines < 0:
        raise ValueError("skip_lines must be non-negative.")

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if not text:
        return []
    if text.endswith("\n"):
        text = text[:-1]
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    return lines[skip_lines:]


def inspect_local_dataset(dataset_dir: str | Path) -> LocalDatasetSummary:
    """Return counts and sparse-matrix shapes for a local dataset folder."""

    from scipy import sparse

    root = Path(dataset_dir)
    required = ["train_texts.txt", "test_texts.txt", "y_train.npz", "y_test.npz"]
    missing = tuple(name for name in required if not (root / name).exists())

    train_text_lines = count_text_lines(root / "train_texts.txt") if (root / "train_texts.txt").exists() else None
    test_text_lines = count_text_lines(root / "test_texts.txt") if (root / "test_texts.txt").exists() else None

    y_train_shape = None
    y_test_shape = None
    y_train_nnz = None
    y_test_nnz = None
    if (root / "y_train.npz").exists():
        y_train = sparse.load_npz(root / "y_train.npz")
        y_train_shape = tuple(int(v) for v in y_train.shape)
        y_train_nnz = int(y_train.nnz)
    if (root / "y_test.npz").exists():
        y_test = sparse.load_npz(root / "y_test.npz")
        y_test_shape = tuple(int(v) for v in y_test.shape)
        y_test_nnz = int(y_test.nnz)

    labels_file = root / "labels.txt"
    label_lines = count_text_lines(labels_file) if labels_file.exists() else None

    return LocalDatasetSummary(
        dataset_dir=root,
        train_text_lines=train_text_lines,
        test_text_lines=test_text_lines,
        y_train_shape=y_train_shape,
        y_test_shape=y_test_shape,
        y_train_nnz=y_train_nnz,
        y_test_nnz=y_test_nnz,
        label_lines=label_lines,
        missing_files=missing,
    )


def load_local_dataset(
    dataset_dir: str | Path,
    *,
    skip_train_lines: int = 0,
    skip_test_lines: int = 0,
) -> LocalDataset:
    """Load a dataset folder containing texts and SciPy sparse label matrices."""

    from scipy import sparse

    root = Path(dataset_dir)
    required = ["train_texts.txt", "test_texts.txt", "y_train.npz", "y_test.npz"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Dataset folder is missing: {joined}")

    train_texts = read_text_lines(root / "train_texts.txt", skip_lines=skip_train_lines)
    test_texts = read_text_lines(root / "test_texts.txt", skip_lines=skip_test_lines)
    y_train = sparse.load_npz(root / "y_train.npz").tocsr()
    y_test = sparse.load_npz(root / "y_test.npz").tocsr()

    if len(train_texts) != y_train.shape[0]:
        raise ValueError(
            "train_texts.txt line count must match y_train rows. "
            f"Got {len(train_texts)} text lines and {y_train.shape[0]} matrix rows. "
            f"Skipped {skip_train_lines} initial train lines. "
            "Check whether the text file has a header, missing documents, or multiline documents."
        )
    if len(test_texts) != y_test.shape[0]:
        raise ValueError(
            "test_texts.txt line count must match y_test rows. "
            f"Got {len(test_texts)} text lines and {y_test.shape[0]} matrix rows. "
            f"Skipped {skip_test_lines} initial test lines. "
            "Check whether the text file has a header, missing documents, or multiline documents."
        )
    if y_train.shape[1] != y_test.shape[1]:
        hint = ""
        if y_test.shape[1] == 0 and y_train.shape[1] > 0:
            hint = (
                " y_test.npz has zero label columns, so it was probably generated "
                "without the training label vocabulary. Recreate y_test.npz with "
                f"shape (n_test, {y_train.shape[1]}) using the same label-id mapping as y_train."
            )
        raise ValueError(
            "y_train and y_test must have the same number of labels. "
            f"Got {y_train.shape[1]} and {y_test.shape[1]} labels."
            f"{hint}"
        )

    labels_file = root / "labels.txt"
    label_names = read_text_lines(labels_file) if labels_file.exists() else None
    if label_names is not None and len(label_names) != y_train.shape[1]:
        raise ValueError(
            "labels.txt line count must match number of labels. "
            f"Got {len(label_names)} label names and {y_train.shape[1]} label columns."
        )

    return LocalDataset(
        train_texts=train_texts,
        test_texts=test_texts,
        y_train=y_train,
        y_test=y_test,
        label_names=label_names,
    )
