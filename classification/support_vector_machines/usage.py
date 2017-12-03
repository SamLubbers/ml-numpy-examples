"""example usage of the smo algorithm"""
import pandas as pd
dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from smo import smo_simple

C = 0.6
tolerance = 0.001
iterations = 40
alphas, bias = smo_simple(X, y, C, tolerance, iterations)

