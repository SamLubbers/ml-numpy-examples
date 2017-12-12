"""example use of functions of the regression_tree module"""

# loading dataset
import pandas as pd
dataset = pd.read_csv('ex00.txt', delimiter='\t', header=None)
dataset.columns = ['x', 'y']

# making binary split according to value
from regression_tree import binary_split
left_split, right_split = binary_split(dataset, 'x', 0.5)