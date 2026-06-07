from __future__ import annotations

import argparse
from pathlib import Path

from xmtc_robustness.data import inspect_local_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local XMTC dataset files.")
    parser.add_argument("--data", required=True, type=Path, help="Dataset folder to inspect.")
    parser.add_argument("--skip-train-lines", default=0, type=int, help="Initial train text lines to ignore in the report.")
    parser.add_argument("--skip-test-lines", default=0, type=int, help="Initial test text lines to ignore in the report.")
    return parser.parse_args()


def status_line(name: str, ok: bool, detail: str) -> str:
    status = "OK" if ok else "FAIL"
    return f"[{status}] {name}: {detail}"


def main() -> None:
    args = parse_args()
    summary = inspect_local_dataset(args.data)

    print(f"Dataset: {summary.dataset_dir}")
    if summary.missing_files:
        print(status_line("required files", False, ", ".join(summary.missing_files)))
        raise SystemExit(1)

    print(f"train_texts.txt lines: {summary.train_text_lines}")
    print(f"test_texts.txt lines:  {summary.test_text_lines}")
    print(f"y_train.npz shape:     {summary.y_train_shape}, nnz={summary.y_train_nnz}")
    print(f"y_test.npz shape:      {summary.y_test_shape}, nnz={summary.y_test_nnz}")
    if summary.label_lines is not None:
        print(f"labels.txt lines:      {summary.label_lines}")

    adjusted_train_lines = None if summary.train_text_lines is None else summary.train_text_lines - args.skip_train_lines
    adjusted_test_lines = None if summary.test_text_lines is None else summary.test_text_lines - args.skip_test_lines

    if args.skip_train_lines or args.skip_test_lines:
        print(f"adjusted train lines:  {adjusted_train_lines} after skipping {args.skip_train_lines}")
        print(f"adjusted test lines:   {adjusted_test_lines} after skipping {args.skip_test_lines}")

    train_ok = adjusted_train_lines == summary.y_train_shape[0]
    test_ok = adjusted_test_lines == summary.y_test_shape[0]
    label_ok = summary.y_train_shape[1] == summary.y_test_shape[1]
    names_ok = summary.label_lines is None or summary.label_lines == summary.y_train_shape[1]

    print()
    print(status_line("train rows", train_ok, "train_texts.txt lines must equal y_train rows"))
    print(status_line("test rows", test_ok, "test_texts.txt lines must equal y_test rows"))
    print(status_line("label columns", label_ok, "y_train and y_test must have the same number of columns"))
    print(status_line("label names", names_ok, "labels.txt is optional, but if present must match label columns"))

    if summary.y_test_shape[1] == 0 and summary.y_train_shape[1] > 0:
        print()
        print("Diagnosis: y_test.npz has zero label columns.")
        print(
            "Regenerate y_test.npz with the same label-id vocabulary used by y_train.npz; "
            f"the expected number of columns is {summary.y_train_shape[1]}."
        )

    if not (train_ok and test_ok and label_ok and names_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
