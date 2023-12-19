import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn(k, point, data):
    distances = np.array([euclidean_distance(point, d) for d in data])
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k]

# Sample Data
data = np.array([[1, 2], [2, 3], [3, 4], [5, 5], [1, 4], [2, 5]])

# Test Point
test_point = np.array([3, 3])

# Number of Neighbors
k = 3

# Find the k-nearest neighbors
neighbors = knn(k, test_point, data)

# Output the indices of the nearest neighbors
print("Indices of the k-nearest neighbors:", neighbors)
