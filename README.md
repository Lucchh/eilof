# EILOF: An Efficient Incremental Local Outlier Factor Algorithm for Data Streaming

[![MIT License](https://img.shields.io/badge/license-MIT-green)][MIT License]

EILOF is a Python package for outlier detection in data streams using an optimized incremental implementation of the Local Outlier Factor (LOF). It is based on the research paper “An Efficient Outlier Detection Algorithm for Data Streaming” (publication forthcoming).

This package offers robust and scalable anomaly detection for real-time data analysis.

---
## Table of Contents

- [Features](#features-section)
- [Installation](#installation-section)
- [Documentation](#documentation-section)
- [Getting Started](#getting-started-section)
- [Incremental Demo](#quick-start-incremental-demo)
- [Dataset Example](#dataset-example-section)
- [Contributing](#contributing-section)
- [Issues](#issues-section)
- [License](#license-section)

---

<h2 id="features-section">✨ Features</h2>

- **Incremental Updates**: Efficiently updates LOF scores when new data points are streamed.  
- **High Performance**: Optimized for computational efficiency in large-scale datasets.  
- **Customizable Parameters**: Adjust the number of neighbors (k) for outlier detection.  
- **Ease of Use**: Intuitive API with minimal setup.
- **Typical Use-Case**: EILOF is especially suited for IoT data, streaming logs, or any scenario where new data arrives continuously, and retraining from scratch is impractical.

---

<h2 id="installation-section">🔧 Installation</h2>

Install EILOF via pip:

```python
pip install eilof
```


---

<h2 id="documentation-section">📚 Documentation</h2>

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
<h2 id="getting-started-section">📖 Getting Started</h2>

Below is a quick overview showing how to initialize and use **EILOF** on a static dataset. However, the real power of EILOF lies in its **incremental update** functionality—check out the [Quick Start: Incremental Demo](#quick-start-incremental-demo).

```python
import numpy as np
from eilof import EILOF

# Create some reference data
reference_data = np.random.rand(100, 2)

# Initialize and fit the model
model = EILOF(k=5)
model.fit(reference_data)

# Predict outliers on the reference data
labels = model.predict_labels(threshold=95)
print("Reference Outlier Labels:", labels)
```

<h2 id="quick-start-incremental-demo"> 🚀 Incremental Demo</h2>

```python
import numpy as np
from eilof import EILOF

# Suppose you have some initial reference data
reference_data = np.random.rand(100, 2)

# Initialize and fit the model on the reference data
model = EILOF(k=5)
model.fit(reference_data)

# Simulate new streaming data arriving in small batches
new_batch_1 = np.random.rand(5, 2)
new_batch_2 = np.random.rand(3, 2)

# Update the model with new streaming data
model.update(new_batch_1)
model.update(new_batch_2)

# Predict outlier labels after incremental updates
all_labels = model.predict_labels(threshold=70)
print("All Data Labels (Reference + New):", all_labels)

# Optionally, get only the new points' labels
new_labels = model.predict_labels(threshold=95, include_reference=False)
print("New Points' Labels:", new_labels)
```


With EILOF, the update() method efficiently adjusts LOF scores to reflect newly arrived points—making it ideal for:
- Resource-constrained environments
- Large-scale streaming scenarios
- Time-sensitive applications
    
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
     ```python
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
     ```python
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
     ```python
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
     ```python
     from eilof import lof_srs

     lof_scores = lof_srs(dist_matrix, neighbors, lrd, 3)
     print("LOF Scores:", lof_scores)
     ```

5. **compute_distances(data, new_point)**
   - **Description**:  
     Computes the Euclidean distances of a new data point from all points in an existing dataset.
   - **Parameters**:
     - `data (numpy.ndarray)`: Existing dataset.
     - `new_point (numpy.ndarray)`: New data point to compute distances for.
   - **Returns**:
     - `distances (numpy.ndarray)`: Array of distances between the new point and each point in the dataset.
   - **Example**:
     ```python
    import numpy as np
    from eilof import compute_distances

    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])  

    new_point = np.array([2.0, 3.0, 4.0]) 

    # Calculate distances
    distances = compute_distances(data, new_point)
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
     ```python
     from eilof import local_outlier_factor

     lof_scores = local_outlier_factor(data, 2)
     print("LOF Scores:", lof_scores)
     ```

#### Example: Combining Multiple Utility Functions

```python
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

<h2 id="dataset-example-section">📊 Dataset Example</h2>


**Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

```python
# Ensure you have Kaggle API credentials set up for automatic dataset download
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from eilof import EILOF
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = os.path.join(path, "creditcard.csv")
data = pd.read_csv(csv_file)

# Preprocess data
scaler = StandardScaler()
features = scaler.fit_transform(data.drop(columns=['Class']))
labels = data['Class']

# Split into reference and streaming datasets
reference_data = features[:500]
streaming_data = features[500:510]

# Initialize and fit the EILOF model
model = EILOF(k=50)
model.fit(reference_data)

# Update model with streaming data
lof_scores = model.update(streaming_data)
outlier_labels = model.predict_labels(threshold=95)

print("Outlier Labels:", outlier_labels)
```

---

<h2 id="contributing-section">🤝 Contributing</h2>


We welcome contributions to the EILOF package! To contribute:

1. Fork this repository.  
2. Create a feature branch:  
   ```python
   git checkout -b feature-branch-name
   ```
3. Commit your changes:
   ```python
   git commit -m "Add feature XYZ"
   ```
4. Push to your branch:
    ```python
    git push origin feature-branch-name
    ```
5. Create a pull request.

---

<h2 id="issues-section">🐞 Issues</h2>


If you encounter any issues or have feature requests, feel free to submit them [here](https://github.com/Lucchh/eilof/issues).

---
<h2 id="license-section">⚖️ License</h2>


This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for details.


