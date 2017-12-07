import pandas as pd

# loading dataset
dataset = pd.read_csv('ex0.txt', delimiter='\t', header=None)

dataset.columns = ['offset', 'x', 'y']

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from linear_regression import calculate_regression_weights

weights = calculate_regression_weights(X, y)

from linear_regression import predict_value

y_hat = predict_value(X, y, X)