"""example usage of the svd module"""
import numpy as np

# create dataset where both features are the same
duplicate_feature_data = np.array([[1, 1], [7, 7]])

from svd import singular_values
sv_dfd = singular_values(duplicate_feature_data)

dataset = np.array([[1, 1, 1, 0, 0],
                    [2, 2, 2, 0, 0],
                    [1, 1, 1, 0, 0],
                    [5, 5, 5, 0, 0],
                    [1, 1, 0, 2, 2],
                    [0, 0, 0, 3, 3],
                    [0, 0, 0, 1, 1]])

from svd import compress_dataset
compressed_dataset = compress_dataset(np.mat(dataset))
