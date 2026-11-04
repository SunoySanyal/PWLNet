"""Reproducible train/validation/test splitting.

The helpers here split a single dataset (a :class:`pandas.DataFrame` or a
NumPy array) into disjoint partitions, returning objects of the same
type so notebooks and benchmarks can keep using column names or array
indexing. Splitting is deterministic given ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DataSplit",
    "train_test_split",
    "train_val_test_split",
]


@dataclass(frozen=True)
class DataSplit:
    """Train/validation/test partitions of one dataset.

    Attributes
    ----------
    train, validation, test : object
        Partitions of the same type as the input (``DataFrame`` or
        ``numpy.ndarray``). ``validation`` is ``None`` when
        ``val_size == 0``.
    """

    train: Any
    validation: Any
    test: Any

    def sizes(self) -> dict[str, int]:
        """Number of rows in each partition (validation counted as 0 if absent)."""
        return {
            "train": len(self.train),
            "validation": 0 if self.validation is None else len(self.validation),
            "test": len(self.test),
        }


def _take(data: Any, index: np.ndarray) -> Any:
    """Select rows ``index`` from a DataFrame (``.iloc``) or array."""
    if hasattr(data, "iloc"):
        return data.iloc[index]
    return np.asarray(data)[index]


def _partition_indices(
    n: int,
    val_size: float,
    test_size: float,
    random_state: int | None,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint (train, val, test) row-index arrays covering ``range(n)``.

    Sizes use floor rounding for test then validation, so the train
    partition absorbs the remainder and every row is assigned exactly
    once.
    """
    if n < 1:
        raise ValueError("Cannot split an empty dataset.")
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size!r}.")
    if not 0.0 <= test_size < 1.0:
        raise ValueError(f"test_size must be in [0, 1), got {test_size!r}.")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must be < 1, got {val_size + test_size!r}."
        )

    order = np.arange(n)
    if shuffle:
        np.random.default_rng(random_state).shuffle(order)

    n_test = int(np.floor(test_size * n))
    n_val = int(np.floor(val_size * n))
    if test_size > 0.0:
        n_test = max(1, n_test)
    if val_size > 0.0:
        n_val = max(1, n_val)
    if n_test + n_val >= n:
        raise ValueError(
            f"Requested validation+test rows ({n_val}+{n_test}) leave no "
            f"training rows for n={n}; reduce val_size/test_size."
        )

    test_idx = order[:n_test]
    val_idx = order[n_test : n_test + n_val]
    train_idx = order[n_test + n_val :]
    return train_idx, val_idx, test_idx


def train_val_test_split(
    data: Any,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> DataSplit:
    """Split one dataset into train/validation/test partitions.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Dataset to split along its first axis.
    val_size, test_size : float, optional
        Fractions of the full dataset assigned to the validation and
        test partitions. Each must be in ``[0, 1)`` and together sum to
        less than 1. Default ``0.2`` each (a 60/20/20 split).
    random_state : int or None, optional
        Seed for the shuffling permutation. Use the same value across
        datasets/notebooks to get identical splits.
    shuffle : bool, optional
        Shuffle before splitting. Default True. With ``shuffle=False``
        the partitions are contiguous slices (test first, then
        validation, then train).

    Returns
    -------
    DataSplit
        Partitions of the same type as ``data``; ``validation`` is
        ``None`` when ``val_size == 0``.
    """
    train_idx, val_idx, test_idx = _partition_indices(
        len(data), val_size, test_size, random_state, shuffle
    )
    return DataSplit(
        train=_take(data, train_idx),
        validation=_take(data, val_idx) if val_idx.size else None,
        test=_take(data, test_idx),
    )


def train_test_split(
    data: Any,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[Any, Any]:
    """Split one dataset into train/test partitions (no validation set).

    Convenience wrapper over :func:`train_val_test_split` with
    ``val_size=0``.

    Returns
    -------
    tuple
        ``(train, test)`` of the same type as ``data``.
    """
    split = train_val_test_split(
        data,
        val_size=0.0,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return split.train, split.test
