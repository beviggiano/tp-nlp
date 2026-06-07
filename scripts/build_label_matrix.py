from __future__ import annotations

import argparse
from pathlib import Path

from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SciPy CSR label matrix from a text file with one list of "
            "integer label ids per instance."
        )
    )
    parser.add_argument("--labels", required=True, type=Path, help="Input label file, one instance per line.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz path.")
    parser.add_argument("--n-labels", required=True, type=int, help="Total number of label columns.")
    parser.add_argument("--skip-lines", default=0, type=int, help="Initial lines to skip, useful for headers.")
    parser.add_argument(
        "--separator",
        default="auto",
        choices=["auto", "comma", "space"],
        help="How labels are separated on each line.",
    )
    parser.add_argument(
        "--allow-out-of-vocabulary",
        action="store_true",
        help="Ignore label ids outside [0, n_labels) instead of failing.",
    )
    return parser.parse_args()


def parse_label_line(line: str, separator: str) -> list[int]:
    line = line.strip()
    if not line:
        return []
    if separator == "comma" or (separator == "auto" and "," in line):
        tokens = [token.strip() for token in line.split(",")]
    else:
        tokens = line.split()
    return [int(token) for token in tokens if token]


def main() -> None:
    args = parse_args()
    if args.n_labels <= 0:
        raise ValueError("--n-labels must be positive.")
    if args.skip_lines < 0:
        raise ValueError("--skip-lines must be non-negative.")

    rows: list[int] = []
    cols: list[int] = []
    n_rows = 0
    skipped_oov = 0

    with args.labels.open("r", encoding="utf-8") as handle:
        for raw_row, line in enumerate(handle):
            if raw_row < args.skip_lines:
                continue
            row = raw_row - args.skip_lines
            n_rows += 1
            for label_id in parse_label_line(line, args.separator):
                if 0 <= label_id < args.n_labels:
                    rows.append(row)
                    cols.append(label_id)
                elif args.allow_out_of_vocabulary:
                    skipped_oov += 1
                else:
                    raise ValueError(
                        f"Label id {label_id} on input line {raw_row + 1} is outside "
                        f"the valid range [0, {args.n_labels})."
                    )

    data = [1] * len(rows)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, args.n_labels), dtype="float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output, matrix)

    print(f"saved: {args.output}")
    print(f"shape: {matrix.shape}")
    print(f"nnz:   {matrix.nnz}")
    if skipped_oov:
        print(f"ignored out-of-vocabulary labels: {skipped_oov}")


if __name__ == "__main__":
    main()
