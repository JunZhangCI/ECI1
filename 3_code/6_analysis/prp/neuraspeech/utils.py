# utils.py
from __future__ import annotations

import os
import copy
import eelbrain
from typing import Dict, List, Optional, Sequence, TypeVar

T = TypeVar('T')

def parse_bids(filename: str) -> Dict[str, str]:
    """
    Parse BIDS-style filename into a dictionary of key-value pairs.

    Example:
    --------
    Input:  'sub-01_task-alice_cond-noise.mat'
    Output: {'sub': '01', 'task': 'alice', 'cond': 'noise'}
    """
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]  # Remove .mat or other extension
    parts = base.split('_')

    metadata = {}
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)
            metadata[key] = value
    return metadata

def combine(datasets: Sequence[T],
            labels: Optional[Sequence[str]] = None,
            label_col: str = 'dataset',
            dim_intersection: bool = False
            ) -> T:
    """
    Combine multiple DatasetBase (or subclasses like PRPData/FStatistic) objects
    by concatenating their underlying eelbrain.Dataset rows.

    Parameters:
    - datasets: Sequence of DatasetBase-like objects (must implement .get_data() and have .data attribute).
    - labels: Optional sequence of strings (same length as `datasets`). If provided, a new column `label_col` is added indicating which source dataset each row came from.
    - label_col: Name of the label column to add when `labels` is not None.
    - dim_intersection: Passed to eelbrain.combine(...). If True, discard NDVar dimensions not present in all datasets (useful when some NDVars have mismatching dims).

    Returns:
    - A deep-copied new object of the same class as datasets[0], with its `.data` replaced by the combined eelbrain.Dataset.
    """
    if not datasets:
        raise ValueError('combine(): `datasets` is empty.')

    if labels is not None and len(labels) != len(datasets):
        raise ValueError(
            f'combine(): `labels` must have the same length as `datasets` '
            f'({len(labels)} vs {len(datasets)}).'
        )

    # Make deep copies of the objects so we never touch the originals
    ds_copies: List[T] = [copy.deepcopy(d) for d in datasets]

    # Extract eelbrain.Dataset copies (your get_data() already returns a copy)
    eel_ds_list: List[eelbrain.Dataset] = []
    for i, d in enumerate(ds_copies):
        eel_ds = d.get_data()
        if eel_ds is None:
            raise ValueError(f'combine(): datasets[{i}].data is None.')
        eel_ds_list.append(eel_ds)

    # Optionally add a label column to each eelbrain.Dataset before combining
    if labels is not None:
        labeled_list: List[eelbrain.Dataset] = []
        for eel_ds, lab in zip(eel_ds_list, labels):
            tmp = eel_ds.copy()
            tmp[label_col] = eelbrain.Factor([lab] * tmp.n_cases)
            labeled_list.append(tmp)
        eel_ds_list = labeled_list

    # Combine the eelbrain.Dataset objects
    combined = eelbrain.combine(eel_ds_list, dim_intersection=dim_intersection)

    # Return a deepcopy of the first object, with combined data
    out = copy.deepcopy(datasets[0])
    out.data = combined
    return out
