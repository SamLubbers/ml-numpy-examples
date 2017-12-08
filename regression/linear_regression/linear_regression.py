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

def predict_values(test_set, data, labels):
    """predicts the values a test set using the linear regression

    :type test_set: numpy.ndarray ((len(test_set) x n)
    :param test_set: test set on which we want to compute the locally weighted linear regression
    :type data: numpy.ndarray (m x n)
    :param data: training set input features data
    :type target_values: numpy.ndarray (m x 1)
    :param target_values: continuous target values associated to each instance in the training data
    :return: predicted value for each new instance
    """
    weights = calculate_regression_weights(data, labels)
    new_instances_matrix = np.mat(test_set)

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

def lwlr_test(test_set, data, target_values,k=1.0):
    """locally weighted linear regression applied on the complete test set

    :type test_set: numpy.ndarray ((len(test_set) x n)
    :param test_set: test set on which we want to compute the locally weighted linear regression
    :type data: numpy.ndarray (m x n)
    :param data: training set input features data
    :type target_values: numpy.ndarray (m x 1)
    :param target_values: continuous target values associated to each instance in the training data
    :param k: value between 0 and 1 that determines how much to weight nearby points
    :return: predicted values obtained from applying lwlr on each instance of the test set
    """
    num_instances_test_set = test_set.shape[0]
    test_set_predictions = np.zeros(num_instances_test_set)
    for index, instance in enumerate(test_set):
        test_set_predictions[index] = lwlr(instance, data, target_values, k)
    return test_set_predictions

def ridge_regression_weights(data_matrix, target_values_matrix, lam=0.2):
    """alculates the linear regression weights for the input data using ridge regression

    :type data_matrix: numpy.matrix (m x n)
    :param data_matrix: training set input features data
    :type target_values_matrix: numpy.matrix (m x 1)
    :param target_values_matrix: continuous target values associated to each instance in the training data
    :param lam: user defined value that determines how our ridge regression will perform
    :return: weights associated to each variable
    """
    xTx = data_matrix.T * data_matrix
    m, n = data_matrix.shape
    xTx_biased = xTx + np.mat(np.eye(n)) * lam
    if np.linalg.det(xTx_biased) == 0:
        print('matrix is singular, cannot do inverse')
        return

    weights = xTx_biased.I * (data_matrix.T * target_values_matrix)
    return weights

def ridge_weights(data, target_values):
    """calculates the regression weights of the input data according to different lambda values

    :type data: numpy.ndarray (m x n)
    :param data: training set input features data
    :type target_values: numpy.ndarray (m x 1)
    :param target_values: continuous target values associated to each instance in the training data
    :return: matrix of regression weights obtained with different alpha values
    """
    data_matrix = np.mat(data)
    target_matrix = np.mat(target_values)

    # normalize data and target values
    target_mean = np.mean(target_matrix, 0)
    target_matrix = target_matrix - target_mean
    data_mean = np.mean(data_matrix, 0)
    data_variance = np.var(data_matrix, 0)
    data_matrix = (data_matrix - target_mean) / data_variance

    # calculate weights for different lambdas
    iterations = 30
    num_features = data_matrix.shape[1]
    weights_matrix = np.zeros((iterations, num_features))
    for i in range(iterations):
        lam = np.exp(i - 10)
        weights = ridge_regression_weights(data_matrix, target_matrix, lam)
        weights_matrix[i, :] = weights.T
    return weights_matrix
