"""using linear regression to predict the age of the abalone shellfish"""
from linear_regression import predict_values, lwlr_test
import pandas as pd
import numpy as np

# loading dataset
dataset = pd.read_csv('abalone.txt', header=None, delimiter='\t')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

#train test split
train_test_split = 0.2
num_test_instances = round(len(dataset)*(train_test_split))

X_train = X[num_test_instances:,:]
X_test = X[:num_test_instances, :]
y_train = y[num_test_instances:,:]
y_test = y[:num_test_instances,:]

# find the k value that yield the lowest error
k_values = [0.1, 1, 10]

min_error = np.inf
best_k_value = 0

y_hat = predict_values(X_test, X_train, y_train)

y_hat_lwlr = lwlr_test(X_test, X_train, y_train)