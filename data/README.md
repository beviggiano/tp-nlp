# Local datasets

Put experimental datasets in subfolders under this directory.

Expected format:

```text
data/
  my_dataset/
    train_texts.txt
    test_texts.txt
    y_train.npz
    y_test.npz
    labels.txt        # optional
```

## Files

- `train_texts.txt`: one training document per line.
- `test_texts.txt`: one test document per line.
- `y_train.npz`: SciPy CSR sparse matrix with shape `(n_train, n_labels)`.
- `y_test.npz`: SciPy CSR sparse matrix with shape `(n_test, n_labels)`.
- `labels.txt`: optional, one label name per line. Internally the toolkit uses
  integer label ids, so this file is only for interpretation/reporting.

Create sparse matrices with:

```python
from scipy import sparse

sparse.save_npz("data/my_dataset/y_train.npz", y_train.tocsr())
sparse.save_npz("data/my_dataset/y_test.npz", y_test.tocsr())
```

The row order in `train_texts.txt` must match `y_train`; the row order in
`test_texts.txt` must match `y_test`.
