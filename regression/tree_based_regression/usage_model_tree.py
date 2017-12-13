"""example use of functions of the model_tree module"""

import pandas as pd
# calculate weights for node
from model_tree import leaf_regression_weights
dataset_3 = pd.read_csv('exp2.txt', delimiter='\t', header=None)
dataset_3.columns = ['x', 'y']
weights = leaf_regression_weights(dataset_3)