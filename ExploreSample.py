import json
import numpy as np
import torch




# Load the .json file
json_file_path = 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset/train_data/ground-truth/013NUWYR/013NUWYR_lower.json'
with open(json_file_path, 'r') as f:
    ground_truth = json.load(f)

# Extract labels and instances
labels = np.array(ground_truth['labels'])
instances = np.array(ground_truth['instances'])

# Print distinct labels before mapping
print(f"Distinct labels before mapping: {np.unique(labels)}")

# Map labels to a smaller range
labels = np.where(labels == 0, 0, labels)  # Keep gingiva as 0
labels = np.where((labels >= 11) & (labels <= 27), labels - 10, labels)  # Map upper jaw to 1-17
labels = np.where((labels >= 31) & (labels <= 47), labels - 13, labels)  # Map lower jaw to 18-34

# Print distinct labels after mapping
print(f"Distinct labels after mapping: {np.unique(labels)}")

# Optionally, reduce the number of vertices
# For example, you can sample every nth vertex
n = 2  # Adjust this value based on your needs
reduced_labels = labels[::n]
reduced_instances = instances[::n]

# Convert to PyTorch tensors if needed
labels_tensor = torch.from_numpy(reduced_labels)
instances_tensor = torch.from_numpy(reduced_instances)

# Print the unique labels after reduction
print(f"Distinct labels after reduction: {torch.unique(labels_tensor)}")