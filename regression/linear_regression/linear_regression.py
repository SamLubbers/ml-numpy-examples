import numpy as np


def calculate_regression_weights(data, target_values):
    """calculates the linear regression weights for the input data using the OLS method

    :type data: numpy.ndarray (m x n)
    :param data: training set input features data
    :type target_values: numpy.ndarray (m x 1)
    :param target_values: continuous target values associated to each instance in the training data
    :return: weights for the regression equation of this dataset
    """
    data_matrix = np.mat(data)
    target_values_matrix = np.mat(target_values)

    xTx = data_matrix.T * data_matrix
    if np.linalg.det(xTx) == 0.0:
        print('matrix is singular, cannot do inverse')
        return

    weights = xTx.I * (data_matrix.T * target_values_matrix)
    return weights

def predict_value(data, labels, new_instances):
    """predicts a numeric value using the regression equation

    :type data: numpy.ndarray (m x n)
    :param data: training set input features data
    :type target_values: numpy.ndarray (m x 1)
    :param target_values: continuous target values associated to each instance in the training data
    :type new_instances: numpy.ndarray (len(new_instaces) x n)
    :param new_instances: new instance(s) we want to predict the value of
    :return: predicted value for each new instance
    """
    weights = calculate_regression_weights(data, labels)
    new_instances_matrix = np.mat(new_instances)

    predicted_values = new_instances_matrix * weights
    return predicted_values
