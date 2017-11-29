"""example use of functions in logistic_regression module"""

# load dataset
import pandas as pd
dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None)

dataset.columns = ['x1', 'x2', 'label']

# we add a column of ones by convention
import numpy as np
dataset.insert(0, 'x0', np.ones(len(dataset)))

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values


# calculate optimal weights
from logistic_regression import optimal_weights

weights = optimal_weights(X, y)
