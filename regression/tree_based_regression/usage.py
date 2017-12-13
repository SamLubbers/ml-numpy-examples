"""example use of functions of the regression_tree module"""

# loading dataset
import pandas as pd
dataset = pd.read_csv('ex00.txt', delimiter='\t', header=None)
dataset.columns = ['x', 'y']

# making binary split according to value
from regression_tree import binary_split
left_split, right_split = binary_split(dataset, 'x', 0.5)

# finding best feature, value split
from regression_tree import choose_best_split
feature, value = choose_best_split(dataset)

# creating regression tree
from regression_tree import create_tree
my_tree = create_tree(dataset)

# check if our subnode is a tree
from regression_tree import is_tree
it_is_tree = is_tree(my_tree['left'])

# mean value of tree
from regression_tree import tree_mean_value
tree_mean = tree_mean_value(my_tree)

# construct tree via postpruning
from regression_tree import prune
dataset_2 = pd.read_csv('ex2.txt', delimiter='\t', header=None)
dataset_2.columns = ['x', 'y']
dataset_2_test = pd.read_csv('ex2test.txt', delimiter='\t', header=None)
dataset_2_test.columns = ['x', 'y']
complex_tree = create_tree(dataset_2, min_error_delta=0, min_instances=1)
pruned_tree = prune(complex_tree, dataset_2_test)