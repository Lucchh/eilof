import numpy as np
from eilof import EILOF, reachability_distance
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load Dataset
try:
    import kagglehub  # Ensure you have kagglehub installed
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = os.path.join(path, "creditcard.csv")
    data = pd.read_csv(csv_file)
except ImportError:
    print("kagglehub not installed. Please install or manually download the dataset.")
    exit()

# Data Preprocessing
outliers = data[data['Class'] == 1]
non_outliers = data[data['Class'] == 0]
n_outliers = len(outliers)
required_total = n_outliers / 0.05
required_non_outliers = int(required_total - n_outliers)
non_outliers_sampled = non_outliers.sample(n=required_non_outliers, random_state=42)
subset = pd.concat([outliers, non_outliers_sampled]).sort_index()
data = subset.drop(columns=['Class'])
label = subset["Class"]

# Scaling Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.values)
selected_data = data_scaled[:1000]
selected_labels = label[:1000].reset_index(drop=True).values
unselected_data = data_scaled[1000:2280]
unselected_labels = label[1000:2280].reset_index(drop=True).values
new_points = unselected_data[:100]

# Step 1: Initialization Test
model = EILOF(k=100)
print(f"Model initialized with k = {model.k}")

# Step 2: Fit Test
model.fit(selected_data)
assert model.lrd is not None, "LRD values should not be None after fitting."
print("Model fit successful. LRD values calculated.")

# Step 3: Update Test
lof_scores = model.update(new_points)
assert lof_scores is not None, "LOF scores should not be None after update."
print("Model updated successfully. Updated LOF scores calculated.")

# Step 4: Prediction Tests
lof_labels = model.predict_labels(threshold=95)
lof_labels_ref = model.predict_reference_labels(threshold=95)
print(f"LOF Labels for new data: {lof_labels}")
print(f"LOF Labels for reference data: {np.where(lof_labels_ref == 1)}")

# Step 5: Utility Function Tests
reach_dist_matrix = reachability_distance(model.dist_matrix, model.k)
assert reach_dist_matrix.shape == model.dist_matrix.shape, "Reachability distance matrix dimensions mismatch."
print("Reachability distance matrix calculated successfully.")
print(f"{reach_dist_matrix}")


# Additional Verifications
print("Model reachability distance for point 950:", model.reachability_dist[950, :][:10])
print("Model LRD values (first 10):", model.lrd[:10])