import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config


def compute_mp(ts: np.ndarray, m: int):
    """
    Compute the matrix profile for self-similarity analysis of a time series.

    Parameters
    ----------
    ts : np.ndarray
        The input time series.
    m : int
        The subsequence length.

    Returns
    -------
    dict
        A dictionary containing the matrix profile, indices, subsequence length, exclusion zone, and input time series.
    """
    # Calculate the exclusion zone internally, based on m
    exclusion_zone = int(np.ceil(m / 2))

    # Compute the matrix profile using STUMP for self-similarity
    mp = stumpy.stump(ts, m)

    return {
        'mp': mp[:, 0],           # Matrix profile values
        'mpi': mp[:, 1],          # Matrix profile indices
        'm': m,                   # Subsequence length
        'excl_zone': exclusion_zone,  # Calculated exclusion zone
        'data': {'ts': ts}        # Original time series data
    }

def meter_swapping_detection(heads, tails, house_idx, m):
    min_score = float('inf')
    min_i, min_j, mp_j = None, None, None
    
    for i in house_idx:
        for j in house_idx:
            if i != j:
                head_i = heads[f'H_{i}'].values
                tail_j = tails[f'T_{j}'].values

                result = compute_mp(head_i, tail_j, m)
                swap_score = np.min(result['mp']) / (np.min(result['mp']) + 1e-6)

                if swap_score < min_score:
                    min_score = swap_score
                    min_i, min_j = i, j
                    mp_j = result['mp']

    # Проверка: если пара не найдена, возвращаем значения по умолчанию
    if min_i is None or min_j is None:
        print("Предупреждение: Подходящая пара не найдена.")
        return {'i': -1, 'j': -1, 'mp_j': []}
    
    return {'i': min_i, 'j': min_j, 'mp_j': mp_j}



