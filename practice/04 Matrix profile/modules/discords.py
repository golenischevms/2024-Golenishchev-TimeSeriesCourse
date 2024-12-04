import numpy as np

from modules.utils import *


import numpy as np
from modules.utils import apply_exclusion_zone

def top_k_discords(matrix_profile: np.ndarray, nn_indices: np.ndarray, excl_zone: int, top_k: int = 15) -> dict:
    """
    Find the top-k discords based on matrix profile.

    Parameters
    ----------
    matrix_profile: the matrix profile structure (array of distances)
    nn_indices: indices of the nearest neighbors
    excl_zone: exclusion zone size
    top_k: number of discords to find

    Returns
    -------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []
    
    # Create a copy of the matrix profile for modification
    modified_matrix_profile = matrix_profile.copy()
    
    # Find top-k discords
    for _ in range(top_k):
        # Find the index of the maximum distance in the modified matrix profile
        max_idx = np.argmax(modified_matrix_profile)
        max_dist = modified_matrix_profile[max_idx]
        nn_idx = nn_indices[max_idx]
        
        # Append the current discord's information
        discords_idx.append(max_idx)
        discords_dist.append(max_dist)
        discords_nn_idx.append(nn_idx)
        
        # Apply the exclusion zone to avoid trivial matches around the selected discord
        modified_matrix_profile = apply_exclusion_zone(modified_matrix_profile, max_idx, excl_zone, -np.inf)
    
    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }


