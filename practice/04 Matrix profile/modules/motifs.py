import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3, excl_zone: int = None) -> dict:

    profile_distances = matrix_profile["mp"]
    profile_indices = matrix_profile["mpi"]

    motifs_idx = []
    motifs_dist = []

    inf_value = np.inf

    for _ in range(top_k):
        motif_idx = np.argmin(profile_distances)
        motif_dist = profile_distances[motif_idx]

        motifs_idx.append((motif_idx, profile_indices[motif_idx]))
        motifs_dist.append(motif_dist)

        if excl_zone is not None:
            profile_distances = apply_exclusion_zone(profile_distances, motif_idx, excl_zone, inf_value)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }