"""example use of functions in logistic_regression module"""

# load dataset
import pandas as pd
dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# calculate optimal weights
from logistic_regression import optimal_weights

weights = optimal_weights(X, y)