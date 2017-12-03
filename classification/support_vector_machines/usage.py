"""example usage of smo algorithm and svm classifier"""
import pandas as pd
dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from smo import smo_simple

C = 0.6
tolerance = 0.001
iterations = 40
alphas, bias = smo_simple(X, y, C, tolerance, iterations)

# see which points are support vectors
for instance, label, alpha in zip(X, y, alphas):
    if alpha > 0:
        print('Support Vector:', instance, label)

# calculate hyperplane parameters
from smo import calculate_hyperplane_parameters
w = calculate_hyperplane_parameters(alphas, X, y)

# classifiy new instance
from svm import classify_svm_linear
import numpy as np
label = classify_svm_linear(np.mat(X[0]), w, bias)
