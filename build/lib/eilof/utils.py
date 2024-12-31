

import numpy as np
from scipy.spatial.distance import cdist

def k_nearest_neighbors(dist_matrix, k):
    """Returns indices of k-nearest neighbors for each point, excluding itself."""
    neighbors = np.argpartition(dist_matrix, range(1, k+1), axis=1)[:, 1:k+1]
    return neighbors

def reachability_distance(dist_matrix, k):
    """Compute the reachability distance for each point."""
    kth_distances = np.sort(dist_matrix, axis=1)[:, k]
    reach_dist_matrix = np.maximum(dist_matrix, kth_distances[None, :])  # Broadcast kth_distances for column use
    return reach_dist_matrix

def local_reachability_density(reachability_dist, neighbors):
    """Computes Local Reachability Density (LRD)."""
    neighbor_dists = reachability_dist[np.arange(reachability_dist.shape[0])[:, None], neighbors]
    lrd = 1 / np.mean(neighbor_dists, axis=1)
    return lrd

def lof_srs(dist_matrix, neighbors, lrd, k):
    neighbor_lrd_sums = np.sum(lrd[neighbors], axis=1)
    LOF_list = neighbor_lrd_sums / (k * lrd)
    return LOF_list

def compute_distances(updated_data, new_point):
    """Computes the Euclidean distances of a new point to all existing points in a dataset."""
    distances = np.sqrt(np.sum((updated_data - new_point) ** 2, axis=1))
    return distances

def local_outlier_factor(data, k):
    """
    Computes LOF scores for a batch dataset.
    
    Parameters:
    - data (numpy.ndarray): Dataset.
    - k (int): Number of neighbors.
    
    Returns:
    - lof_scores (numpy.ndarray): Array of LOF scores.
    """
    dist_matrix = cdist(data, data, 'euclidean')
    neighbors = k_nearest_neighbors(dist_matrix, k)
    reachability_dist = reachability_distance(dist_matrix, k)
    lrd = local_reachability_density(reachability_dist, neighbors)
    lof_scores = lof_srs(dist_matrix, neighbors, lrd, k)
    return lof_scores

def incremental_lof_update_new(updated_data, k, new_point, reachability_dist, lof_og, lrd_og, dist_matrix):
    """
    Incrementally updates LOF scores for a streaming data point.

    Parameters:
    - updated_data (numpy.ndarray): Current dataset.
    - k (int): Number of nearest neighbors.
    - new_point (numpy.ndarray): New data point.
    - reachability_dist (numpy.ndarray): Reachability distance matrix.
    - lof_og (numpy.ndarray): LOF scores.
    - lrd_og (numpy.ndarray): LRD values.

    Returns:
    - Updated LOF scores (numpy.ndarray).
    - Updated reachability distance matrix (numpy.ndarray).
    - Updated LRD values (numpy.ndarray).
    """
    new_distances = compute_distances(updated_data, new_point)
    dist_matrix = np.pad(dist_matrix, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    dist_matrix[-1, :-1] = new_distances
    dist_matrix[:-1, -1] = new_distances
    updated_data = np.append(updated_data, [new_point], axis=0) #update the data
    # kth distance of the new point
    k_nearest_neib_new = np.sort(new_distances)[k-1]
    # Compute reachability distance from new point to its k-nearest neighbors

    # Compute the sorted indices (k-nearest neighbors) for the new point
    k_nearest_indices_new = np.argsort(dist_matrix)[:, 1:k+1][-1]

    neighbors_new = k_nearest_neighbors(dist_matrix, k)
    kth_distances_new = np.partition(dist_matrix, k, axis=1)[:, k]
    reachability_distance_new = []
    kth_distances = np.partition(dist_matrix, k, axis=1)[:, k]

    reachability_dist = np.pad(reachability_dist, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    lrd_update_list = []
    for i in k_nearest_indices_new:
          reach_dist = max(kth_distances_new[i], new_distances[i])
          reachability_distance_new.append(reach_dist)
          reachability_dist[-1, i] = reach_dist
          if len(updated_data)-2 in neighbors_new[i]:
             lrd_update_list.append(i)
             reachability_dist[i, -1] = max(k_nearest_neib_new, new_distances[i])


    for i in lrd_update_list:
        lrd_og[i] = 1.0 / (sum(reachability_dist[i][neighbors_new[i]])/ k)

    # Compute the LRD for the new point
    lrd_pc = 1.0 / (sum(reachability_distance_new) / k)
    lrd_sum_all = np.append(lrd_og, lrd_pc)

    # Compute the LOF score for the new point
    lof_new = np.mean(lrd_sum_all[k_nearest_indices_new]) / lrd_pc
    lof_og = np.append(lof_og, lof_new)

    return lof_og, reachability_dist, lrd_sum_all, dist_matrix