import numpy as np

def sigmoid(z):
    """applies sigmoid function on z
    :param z: input to the sigmoid function
    :return: number between 0 and 1
    """
    return 1.0/(1.0+np.exp(-z))

def optimal_weights_gradient_ascent(data, labels, alpha=0.001, max_cycles=500):
    """uses gradient ascent to calculate the optimal weight for each feature

    :type data: numpy.ndarray (m, n)
    :param data: set of instances and features
    :type labels: numpy.ndarray (m, 1)
    :param labels: set of labels associated to each instance
    :return: numpy.ndarray of optimal weights for each feature
    """
    data_matrix = np.mat(data)
    labels_matrix = np.mat(labels)
    m, n = data_matrix.shape
    weights = np.ones((n, 1)) # vector containing one weight for each feature
    for _ in range(max_cycles):
        predictions = sigmoid(data_matrix*weights)
        error_rate = (labels_matrix - predictions)
        weights = weights + alpha * data_matrix.transpose() * error_rate

    return weights.getA()

def optimal_weights_stochastic_ascent(data, labels, alpha=0.01):
    """uses stochastic gradient ascent to calculate the optimal weight for each feature

    :type data: numpy.ndarray (m, n)
    :param data: set of instances and features
    :type labels: numpy.ndarray (m, 1)
    :param labels: set of labels associated to each instance
    :return: numpy.ndarray of optimal weights for each feature
    """
    num_features = len(data[0])
    weights = np.ones(num_features)  # vector containing one weight for each feature
    for index, instance in enumerate(data):
        prediction = sigmoid(sum(instance*weights))
        error_rate = (labels[index] - prediction)
        weights = weights + alpha * instance * error_rate

    return weights
