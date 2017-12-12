import pandas as pd

# loading dataset
dataset = pd.read_csv('ex0.txt', delimiter='\t', header=None)

dataset.columns = ['offset', 'x', 'y']

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# linear regression
from linear_regression import calculate_regression_weights

weights = calculate_regression_weights(X, y)

# linear regression on our training data
from linear_regression import predict_values
import numpy as np

X_sorted = np.sort(X.copy(), 0)
y_hat = predict_values(X_sorted, X, y)

# local weighted linear regression on our training data
from linear_regression import lwlr_test
y_hat_lwlr = lwlr_test(X_sorted, X, y, k=0.01)

# ridge regression
dataset_abalone = pd.read_csv('abalone.txt', header=None, delimiter='\t')

X_abalone = dataset_abalone.iloc[:, :-1].values
y_abalone = dataset_abalone.iloc[:, -1:].values

from linear_regression import multiple_ridge_weights
abalone_multiple_ridge_weights = multiple_ridge_weights(X_abalone, y_abalone)

from linear_regression import multiple_stagewise_weights
abalone_multiple_stagewise_weights = multiple_stagewise_weights(X_abalone,
                                                                y_abalone,
                                                                step_size=0.005,
                                                                iterations=1000)