# EILOF: An Efficient Incremental Local Outlier Factor Algorithm for Data Streaming

[![MIT License](https://img.shields.io/badge/license-MIT-green)][MIT License]

EILOF is a Python package for outlier detection in data streams using an optimized incremental implementation of the Local Outlier Factor (LOF). It is based on the research paper “An Efficient Outlier Detection Algorithm for Data Streaming” (publication forthcoming).

This package offers robust and scalable anomaly detection for real-time data analysis.

---
## Table of Contents

## Table of Contents

- [:sparkles: Features](#features)
- [:wrench: Installation](#installation)
- [:book: Getting Started](#getting-started)
- [:books: Documentation](#documentation)
- [:bar_chart: Dataset Example](#dataset-example)
- [:handshake: Contributing](#contributing)
- [:beetle: Issues](#issues)
- [:scales: License](#license)
---

## :sparkles: Features

- **Incremental Updates**: Efficiently updates LOF scores when new data points are streamed.  
- **High Performance**: Optimized for computational efficiency in large-scale datasets.  
- **Customizable Parameters**: Adjust the number of neighbors (k) for outlier detection.  
- **Ease of Use**: Intuitive API with minimal setup.

---

## 🔧 Installation

Install EILOF via pip:

```bash
pip install eilof

```

## 📖 Getting Started

Below is an introduction to the main EILOF class and its usage. For more examples, see the [Documentation](#documentation) section.

---

## 📚 Documentation

### References

- **sklearn.neighbors.LocalOutlierFactor**  
  The Local Outlier Factor (LOF) implementation from scikit-learn is designed for static datasets and is a widely used method for unsupervised outlier detection. While EILOF shares conceptual similarities, it extends LOF to handle incremental updates for streaming data efficiently.  
  Documentation: [scikit-learn LocalOutlierFactor](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)

#### Introduction Example

EILOF builds upon the core concept of Local Outlier Factor (LOF), popularized by `sklearn.neighbors.LocalOutlierFactor`, and adapts it for real-time data streaming scenarios. While the traditional LOF method is well-suited for static datasets, EILOF provides incremental updates, making it ideal for large-scale, real-time applications.

---

### Class: EILOF

This is the primary class of the package that implements the **Efficient Incremental Local Outlier Factor (EILOF)** algorithm.

**Methods**:

1. **fit(data)**
   - **Description**: Fits the model to the initial dataset.
   - **Parameters**:
     - `data (numpy.ndarray)`: A 2D array of data points.
   - **Returns**: None.

2. **update(new_points)**
   - **Description**: Updates the model incrementally with new streaming data points.
   - **Parameters**:
     - `new_points (numpy.ndarray)`: A 2D array of new data points to update the model.
   - **Returns**: Updated LOF scores for the dataset.

3. **predict_labels(threshold=95, include_reference=True)**
   - **Description**: Predicts outlier labels for the dataset based on the specified percentile threshold.
   - **Parameters**:
     - `threshold (float)`: Percentile threshold for detecting outliers.
     - `include_reference (bool)`: Whether to include the reference dataset in the predictions.
   - **Returns**: Binary array of outlier labels (1 = outlier, 0 = inlier).

4. **predict_reference_labels(threshold=95)**
   - **Description**: Predicts outlier labels for the reference dataset.
   - **Parameters**:
     - `threshold (float)`: Percentile threshold for detecting outliers.
   - **Returns**: Binary array of outlier labels for the reference data.

---

### Utility Functions

These utility functions are part of the EILOF package, designed to provide advanced users with flexibility and insights into the Local Outlier Factor (LOF) algorithm.

1. **k_nearest_neighbors(dist_matrix, k)**
   - **Description**:  
     Returns the indices of the k nearest neighbors for each point in a distance matrix, excluding the point itself.
   - **Parameters**:
     - `dist_matrix (numpy.ndarray)`: Pairwise distance matrix.
     - `k (int)`: Number of nearest neighbors to retrieve.
   - **Returns**:
     - `neighbors (numpy.ndarray)`: Indices of the k nearest neighbors for each point.
   - **Example**:
     ```bash
     import numpy as np
     from eilof import k_nearest_neighbors

     dist_matrix = np.random.rand(5, 5)  # Example distance matrix
     neighbors = k_nearest_neighbors(dist_matrix, 3)
     print("Nearest neighbors:", neighbors)
     ```

2. **reachability_distance(dist_matrix, k)**
   - **Description**:  
     Computes the reachability distance matrix for a dataset based on its pairwise distance matrix.
   - **Parameters**:
     - `dist_matrix (numpy.ndarray)`: Pairwise distance matrix.
     - `k (int)`: Number of nearest neighbors.
   - **Returns**:
     - `reach_dist_matrix (numpy.ndarray)`: Reachability distance matrix.
   - **Example**:
     ```bash
     from eilof import reachability_distance

     reach_dist = reachability_distance(dist_matrix, 3)
     print("Reachability Distance Matrix:", reach_dist)
     ```

3. **local_reachability_density(reachability_dist, neighbors)**
   - **Description**:  
     Computes the Local Reachability Density (LRD) for each point, which measures how “densely” a point is located with respect to its neighbors.
   - **Parameters**:
     - `reachability_dist (numpy.ndarray)`: Reachability distance matrix.
     - `neighbors (numpy.ndarray)`: Indices of the k-nearest neighbors for each point.
   - **Returns**:
     - `lrd (numpy.ndarray)`: Local Reachability Density values for all points.
   - **Example**:
     ```bash
     from eilof import local_reachability_density

     lrd = local_reachability_density(reach_dist, neighbors)
     print("Local Reachability Density:", lrd)
     ```

4. **lof_srs(dist_matrix, neighbors, lrd, k)**
   - **Description**:  
     Computes the Local Outlier Factor (LOF) scores for all points, which measure the degree to which a point is an outlier.
   - **Parameters**:
     - `dist_matrix (numpy.ndarray)`: Pairwise distance matrix.
     - `neighbors (numpy.ndarray)`: Indices of the k-nearest neighbors for each point.
     - `lrd (numpy.ndarray)`: Local reachability density values.
     - `k (int)`: Number of nearest neighbors.
   - **Returns**:
     - `LOF_list (numpy.ndarray)`: LOF scores for all points.
   - **Example**:
     ```bash
     from eilof import lof_srs

     lof_scores = lof_srs(dist_matrix, neighbors, lrd, 3)
     print("LOF Scores:", lof_scores)
     ```

5. **compute_distances(updated_data, new_point)**
   - **Description**:  
     Computes the Euclidean distances of a new data point from all points in an existing dataset.
   - **Parameters**:
     - `updated_data (numpy.ndarray)`: Existing dataset.
     - `new_point (numpy.ndarray)`: New data point to compute distances for.
   - **Returns**:
     - `distances (numpy.ndarray)`: Array of distances between the new point and each point in the dataset.
   - **Example**:
     ```bash
     from eilof import compute_distances

     distances = compute_distances(updated_data, new_point)
     print("Distances:", distances)
     ```

6. **local_outlier_factor(data, k)**
   - **Description**:  
     Computes Local Outlier Factor (LOF) scores for a batch dataset.
   - **Parameters**:
     - `data (numpy.ndarray)`: Dataset.
     - `k (int)`: Number of nearest neighbors.
   - **Returns**:
     - `lof_scores (numpy.ndarray)`: Array of LOF scores.
   - **Example**:
     ```bash
     from eilof import local_outlier_factor

     lof_scores = local_outlier_factor(data, 3)
     print("LOF Scores:", lof_scores)
     ```

#### Example: Combining Multiple Utility Functions

```bash
import numpy as np
from eilof import (
    k_nearest_neighbors,
    reachability_distance,
    local_reachability_density,
    lof_srs,
)

data = np.random.rand(10, 3)  # Example dataset
dist_matrix = np.linalg.norm(data[:, None] - data, axis=2)  # Pairwise distances
k = 3

# Compute utilities
neighbors = k_nearest_neighbors(dist_matrix, k)
reach_dist = reachability_distance(dist_matrix, k)
lrd = local_reachability_density(reach_dist, neighbors)
lof_scores = lof_srs(dist_matrix, neighbors, lrd, k)

print("LOF Scores:", lof_scores)
```

## 📊 Dataset Example

**Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

```bash
import pandas as pd
from sklearn.preprocessing import StandardScaler
from eilof import EILOF

# Load dataset
data = pd.read_csv('creditcard.csv')

# Preprocess data
scaler = StandardScaler()
features = scaler.fit_transform(data.drop(columns=['Class']))
labels = data['Class']

# Split into reference and streaming datasets
reference_data = features[:2000]
streaming_data = features[2000:3000]

# Initialize and fit the EILOF model
model = EILOF(k=50)
model.fit(reference_data)

# Update model with streaming data
lof_scores = model.update(streaming_data)
outlier_labels = model.predict_labels(threshold=95)

print("Outlier Labels:", outlier_labels)
```

---

## **🤝 Contributing**

We welcome contributions to the EILOF package! To contribute:

1. Fork this repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature-branch-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature XYZ"
   ```
4. Push to your branch:
    ```bash
    git push origin feature-branch-name
    ```
5. Create a pull request.

---

## 🐞 Issues

If you encounter any issues or have feature requests, feel free to submit them [here](https://github.com/Lucchh/eilof/issues).

---
## **⚖️ License**

This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for details.


