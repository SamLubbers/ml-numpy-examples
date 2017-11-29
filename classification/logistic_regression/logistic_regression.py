import numpy as np
import random
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
    weights = np.ones((n, 1)) # matrix containing one weight for each feature
    for _ in range(max_cycles):
        predictions = sigmoid(data_matrix*weights)
        error_rate = (labels_matrix - predictions)
        weights = weights + alpha * data_matrix.transpose() * error_rate

    return weights.getA()

def optimal_weights_stochastic_ascent(data, labels, num_iter=150):
    """uses stochastic gradient ascent to calculate the optimal weight for each feature

    :type data: numpy.ndarray (m, n)
    :param data: set of instances and features
    :type labels: numpy.ndarray (m, 1)
    :param labels: set of labels associated to each instance
    :return: numpy.ndarray of optimal weights for each feature
    """
    alpha_start_value = 4
    alpha_constant = 0.01

    m, n = data.shape
    weights = np.ones(n)  # vector containing one weight for each feature
    for j in range(num_iter):
        unaccessed_indexes = list(range(m))
        for i in range(m):
            alpha = alpha_start_value/(1.0+j+i) + alpha_constant # step size decreases as iterations increase
            random_index = int(random.uniform(0, len(unaccessed_indexes)))
            prediction = sigmoid(sum(data[random_index]*weights))
            error_rate = (labels[random_index] - prediction)
            weights = weights + alpha * data[random_index] * error_rate
            del(unaccessed_indexes[random_index])

    return weights


def classify_logistic_regression(new_vector, weights):
    """classifies a new instance using the sigmoid function

    :param new_vector: vector of features that we want to classify
    :param weights: optimal weights associated to each feature in the vector
    :return: predicted class label (0 or 1)
    """
    prob = sigmoid(sum(new_vector*weights))
    return 0 if prob < 0.5 else 1