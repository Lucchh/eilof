# import numpy as np
# from scipy.spatial.distance import cdist

# def k_nearest_neighbors(dist_matrix, k):
#     """Returns indices of k-nearest neighbors for each point, excluding itself."""
#     neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
#     return neighbors

# def reachability_distance(dist_matrix, k):
#     """Compute the reachability distance for each point."""
#     kth_distances = np.sort(dist_matrix, axis=1)[:, k]
#     reach_dist_matrix = np.maximum(dist_matrix, kth_distances[None, :])  # Broadcast kth_distances for column use
#     return reach_dist_matrix

# def local_reachability_density(reachability_dist, neighbors):
#     """Computes Local Reachability Density (LRD)."""
#     lrd = 1 / np.mean(reachability_dist[np.arange(len(reachability_dist))[:, None], neighbors], axis=1)
#     return lrd

# def lof_srs(dist_matrix, neighbors, lrd, k):
#     num_points = len(lrd)
#     LOF_list = np.zeros(num_points)
#     for i in range(num_points):
#         lof_sum = 0
#         for neighbor in neighbors[i]:
#             lof_sum += lrd[neighbor]
#         LOF_list[i] = (lof_sum / k) / lrd[i]
#     return LOF_list

# def compute_distances(updated_data, new_point):
#     """Computes the Euclidean distances of a new point to all existing points in a dataset."""
#     distances = np.sqrt(np.sum((updated_data - new_point) ** 2, axis=1))
#     return distances

# def local_outlier_factor(data, k):
#     """
#     Computes LOF scores for a batch dataset.
    
#     Parameters:
#     - data (numpy.ndarray): Dataset.
#     - k (int): Number of neighbors.
    
#     Returns:
#     - lof_scores (numpy.ndarray): Array of LOF scores.
#     """
#     dist_matrix = cdist(data, data, 'euclidean')
#     neighbors = k_nearest_neighbors(dist_matrix, k)
#     reachability_dist = reachability_distance(dist_matrix, k)
#     lrd = local_reachability_density(reachability_dist, neighbors)
#     lof_scores = lof_srs(dist_matrix, neighbors, lrd, k)
#     return lof_scores

# def incremental_lof_update_new(updated_data, k, new_point, reachability_dist, lof_og, lrd_og):
#     """
#     Incrementally updates LOF scores for a streaming data point.

#     Parameters:
#     - updated_data (numpy.ndarray): Current dataset.
#     - k (int): Number of nearest neighbors.
#     - new_point (numpy.ndarray): New data point.
#     - reachability_dist (numpy.ndarray): Reachability distance matrix.
#     - lof_og (numpy.ndarray): LOF scores.
#     - lrd_og (numpy.ndarray): LRD values.

#     Returns:
#     - Updated LOF scores (numpy.ndarray).
#     - Updated reachability distance matrix (numpy.ndarray).
#     - Updated LRD values (numpy.ndarray).
#     """
#     distance_new = compute_distances(updated_data, new_point) #calculate the distance of new point to other points

#     updated_data = np.append(updated_data, [new_point], axis=0) #update the data
#     # kth distance of the new point
#     k_nearest_neib_new = np.sort(distance_new)[k-1]
#     # Compute reachability distance from new point to its k-nearest neighbors
#     dist_matrix = cdist(updated_data, updated_data, 'euclidean') #new distance matrix
#     # Compute the sorted indices (k-nearest neighbors) for the new point
#     k_nearest_indices_new = np.argsort(dist_matrix)[:, 1:k+1][-1]

#     neighbors_new = k_nearest_neighbors(dist_matrix, k)
#     kth_distances_new = np.sort(dist_matrix, axis=1)[:, k]
#     reachability_distance_new = []
#     kth_distances = np.sort(dist_matrix, axis=1)[:, k]

#     reachability_dist = np.pad(reachability_dist, ((0, 1), (0, 1)), mode='constant', constant_values=0)

#     lrd_update_list = []
#     for i in k_nearest_indices_new:
#           reach_dist = max(kth_distances_new[i], distance_new[i])
#           reachability_distance_new.append(reach_dist)
#           reachability_dist[-1, i] = reach_dist
#           if len(updated_data)-2 in neighbors_new[i]:
#              lrd_update_list.append(i)
#              reachability_dist[i, -1] = max(k_nearest_neib_new, distance_new[i])


#     for i in lrd_update_list:
#         lrd_og[i] = 1.0 / (sum(reachability_dist[i][neighbors_new[i]])/ k)

#     # Compute the LRD for the new point
#     lrd_pc = 1.0 / (sum(reachability_distance_new) / k)
#     lrd_sum_all = np.append(lrd_og, lrd_pc)

#     # Compute the LOF score for the new point
#     average_lrd_neighbors = sum(lrd_sum_all[neighbor] for neighbor in k_nearest_indices_new) / k
#     lof_new = average_lrd_neighbors / lrd_pc
#     lof_og = np.append(lof_og, lof_new)

#     return lof_og, reachability_dist, lrd_sum_all, dist_matrix


# class EILOF:
#     def __init__(self, k=5):
#         """
#         Initialize the EILOF algorithm.
        
#         Parameters:
#         - k (int): Number of nearest neighbors to consider for LOF computation.
#         """
#         self.k = k
#         self.data = None
#         self.dist_matrix = None
#         self.reachability_dist = None
#         self.lrd = None
#         self.lof_scores = None
#         self.reference_lof_scores = None  # Store LOF scores for the reference data

#     def fit(self, data):
#         """Fit the model with a reference dataset."""
#         self.data = data
#         self.dist_matrix = cdist(data, data, 'euclidean')
#         self.reachability_dist = reachability_distance(self.dist_matrix, self.k)
#         neighbors = k_nearest_neighbors(self.dist_matrix, self.k)
#         self.lrd = local_reachability_density(self.reachability_dist, neighbors)
#         self.lof_scores = lof_srs(self.dist_matrix, neighbors, self.lrd, self.k)

#         # Store reference LOF scores
#         self.reference_lof_scores = self.lof_scores.copy()
#     def update(self, new_points):
#         """
#         Update the model with one or more new data points.

#         Parameters:
#         - new_points (numpy.ndarray): New data points.

#         Returns:
#         - Updated LOF scores.
#         """
#         new_points = np.atleast_2d(new_points)  # Ensure 2D array for new points

#         for new_point in new_points:
#             (
#                 self.lof_scores,
#                 self.reachability_dist,
#                 self.lrd,
#                 self.dist_matrix,
#             ) = incremental_lof_update_new(
#                 self.data, self.k, new_point, self.reachability_dist, self.lof_scores, self.lrd
#             )
#             self.data = np.vstack((self.data, new_point))  # Append new point to data

#         return self.lof_scores
#     def predict_reference_labels(self, threshold=95):
#         """
#         Predict labels for the reference dataset based on the LOF scores and the user-defined threshold.
        
#         Parameters:
#         - threshold (float): Percentile threshold for outlier detection.
        
#         Returns:
#         - reference_labels (numpy.ndarray): Binary array (1 for outlier, 0 for inlier) for reference data.
#         """
#         if self.reference_lof_scores is None:
#             raise ValueError("Model has not been fitted. Call `fit` before predicting reference labels.")

#         # Calculate threshold based on reference LOF scores
#         threshold_value = np.percentile(self.reference_lof_scores, threshold)
#         return (self.reference_lof_scores > threshold_value).astype(int)

#     def predict_labels(self, threshold=95, include_reference=True):
#         """
#         Predict labels for the dataset based on the LOF scores and the user-defined threshold.
        
#         Parameters:
#         - threshold (float): Percentile threshold for outlier detection.
#         - include_reference (bool): Whether to include predictions for the reference dataset.
        
#         Returns:
#         - predicted_labels (numpy.ndarray): Binary array (1 for outlier, 0 for inlier).
#         """
#         if self.lof_scores is None:
#             raise ValueError("Model has not been fitted. Call `fit` before predicting labels.")

#         # Calculate threshold based on LOF scores
#         threshold_value = np.percentile(self.lof_scores, threshold)
#         predicted_labels = (self.lof_scores > threshold_value).astype(int)

#         # Handle inclusion/exclusion of reference points
#         if include_reference:
#             return predicted_labels
#         else:
#             if len(self.reference_lof_scores) == len(self.lof_scores):
#                 return np.array([])  # No new points
#             return predicted_labels[len(self.reference_lof_scores):]