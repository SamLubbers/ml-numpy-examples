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

def lwlr(local_point, data, target_values,k=1.0):
    """locally weighted linear regression on the local_point

    :type local_point: numpy.ndarray (1 x n)
    :param local_point: instance on which we want to compute the locally weighted linear regression
    :type data: numpy.ndarray (m x n)
    :param data: training set input features data
    :type target_values: numpy.ndarray (m x 1)
    :param target_values: continuous target values associated to each instance in the training data
    :param k: value between 0 and 1 that determines how much to weight nearby points
    :return: predicted value of local_point by computing locally weighted linear regression around that point
    """
    data_matrix = np.mat(data)
    target_values_matrix = np.mat(target_values)

    num_instances = data_matrix.shape[0]
    weights = np.mat(np.eye(num_instances))

    for i, x in enumerate(data_matrix):
        diff_matrix = local_point - x
        weights[i, i] = np.exp(diff_matrix * diff_matrix.T/(-2.0*(k**2)))

    xTx = data_matrix.T * (weights * data_matrix)

    if np.linalg.det(xTx) == 0.0:
        print('matrix is singular, cannot do inverse')
        return

    weights_local_regression = xTx.I * (data_matrix.T * (weights * target_values_matrix))
    prediction = local_point * weights_local_regression
    return prediction