import numpy as np


def calculate_regression_weights(data, labels):
    """calculates the linear regression weights for the input data using the OLS method

    :param data: numpy.ndarray (m x n) of training set data
    :param labels: numpy.ndarray (m x 1) containing the labels associated to each instance in the training data
    :return: weights for the regression equation
    """
    data_matrix = np.mat(data)
    labels_matrix = np.mat(labels)

    xTx = data_matrix.T * data_matrix
    if np.linalg.det(xTx) == 0.0:
        print('matrix is singular, cannot do inverse')
        return

    weights = xTx.I * (data_matrix.T * labels_matrix)
    return weights
