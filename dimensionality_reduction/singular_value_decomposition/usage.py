"""example usage of the svd module"""
import numpy as np

# create dataset where both features are the same
duplicate_feature_data = np.array([[1, 1], [7, 7]])

from svd import singular_values
sv_dfd = singular_values(duplicate_feature_data)
