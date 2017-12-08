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