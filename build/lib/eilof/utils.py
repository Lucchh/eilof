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