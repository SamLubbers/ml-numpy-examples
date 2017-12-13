"""example use of functions of the model_tree module"""
import pandas as pd

# calculate weights for node
from model_tree import leaf_regression_weights
dataset = pd.read_csv('exp2.txt', delimiter='\t', header=None)
dataset.columns = ['x', 'y']
weights = leaf_regression_weights(dataset)

# calculate residual sum of squares error for the given dataset
from model_tree import model_error
err = model_error(dataset)

# best split for model tree
from model_tree import choose_best_split
best_split = choose_best_split(dataset)

# build model tree
from model_tree import create_model_tree
my_tree = create_model_tree(dataset, min_error_delta=1, min_instances=10)
