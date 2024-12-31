import numpy as np
from eilof_model import EILOF

import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
import pandas as pd

# Assuming a CSV file is in the dataset directory
csv_file = os.path.join(path, "creditcard.csv")  # Replace with the actual file name
data = pd.read_csv(csv_file)

# Display the first few rows of the dataset
data.head()
import pandas as pd

# Step 1: Identify outliers and non-outliers
outliers = data[data['Class'] == 1]
non_outliers = data[data['Class'] == 0]

# Step 2: Calculate the number of non-outliers required
n_outliers = len(outliers)  # Total number of outliers
required_total = n_outliers / 0.05  # Total rows needed for 5% outliers
required_non_outliers = int(required_total - n_outliers)

# Step 3: Randomly sample the required number of non-outliers
non_outliers_sampled = non_outliers.sample(n=required_non_outliers, random_state=42)

# Step 4: Combine outliers and sampled non-outliers
subset = pd.concat([outliers, non_outliers_sampled])

# Step 5: Maintain time-series order
subset = subset.sort_index()

# Step 6: Verify the outlier ratio
actual_outlier_ratio = len(subset[subset['Class'] == 1]) / len(subset)
print(f"Outlier Ratio in Subset: {actual_outlier_ratio:.2%}")
# Display the subset
subset.head()

from sklearn.preprocessing import StandardScaler
data = subset.drop(columns=['Class'])
label = subset["Class"]

# Initialize selected and unselected datasets
selected_data = data[:1000].reset_index(drop=True).values
selected_labels = label[:1000].reset_index(drop=True).values
unselected_data = data[1000:2280].reset_index(drop=True).values
unselected_labels = label[1000:2280].reset_index(drop=True).values
scaler = StandardScaler()
selected_data = scaler.fit_transform(selected_data)
scaler = StandardScaler()
unselected_data = scaler.fit_transform(unselected_data)
data = selected_data
new_point = unselected_data



# Step 1: Test Initialization
model = EILOF(k=2)
print("Model initialized with k =", model.k)

# Step 2: Test Fitting the Model
model.fit(data)
print("Model fitted. Local Reachability Densities (LRDs):")
print(model.lrd)

lof_scores = model.update(new_point)
print("After updating with a new point:")
print("Updated LOF Scores:", lof_scores)
lof_label = model.predict_labels(threshold=70)
print("Predicted Labels (threshold=70):", lof_label)