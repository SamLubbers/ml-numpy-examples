"""Sequential Minimal Optimization used to figure out the optimal separating hyperplane"""
import random
import numpy as np

def random_index(i, set_size):
    """finds a random index used to access an item in a set (list, numpy.array ...)

    this index must be within the boundaries of the set size and cannot equal index i
    it is used in the smo algorithm to get the index of the second alpha value used in the optimization

    :param i: value which the new index cannot have
    :param set_size: total number of instances in our set
    :return: index of alpha2 value
    """
    j = i
    while(j==i):
        j = int(random.uniform(0, set_size))
    return j

def bound_alpha(alpha, max_limit, min_limit):
    """bounds alpha value between certain limits"""
    if alpha > max_limit:
        alpha = max_limit
    if alpha < min_limit:
        alpha = min_limit

    return alpha

def instance_error_rate(i, data_matrix, labels_matrix, alphas, bias):
    """calculates the error rate of an instance i in our dataset given the current alphas and bias values"""

    prediction_i = float(np.multiply(alphas, labels_matrix).T * (data_matrix * data_matrix[i, :].T)) + bias
    error_i = prediction_i - float(labels_matrix[i])
    return error_i

def can_be_optimized(alpha, label, error, C, tolerance):
    """determines if a certain value of alpha can be optimized"""
    if ((label * error < -tolerance) and (alpha < C)) or ((label * error > tolerance) and (alpha > 0)):
        return True
    return False

def smo_simple(data, labels, C, tolerance, max_iterations):
    """simple implementation of the smo algorithm

    :type data: numpy.ndarray (m, n)
    :param data: set of instances and features
    :type labels: numpy.ndarray (m, 1)
    :param labels: set of labels associated to each instance. Labels must be -1 or 1
    :param C: slack variable, which works as an upper limit for alpha | 0 <= alpha <= C
    :param tolerance: criterion by which we determine if a data vector can be optimized
    :param max_iterations: number of iterations to be carried out in the optimization
    :return: alphas, bias - optimal parameters for the hyperplane separators
    """
    # numpy array to matrix for easier operations
    data_matrix = np.mat(data)
    labels_matrix = np.mat(labels)
    # initialize alphas and bias
    m, n = data_matrix.shape
    alphas = np.zeros((m, 1))
    bias = 0
    # run optimization
    iteration = 0
    while(iteration<max_iterations):
        alpha_pairs_changed = 0
        for i in range(m):
            error_i = instance_error_rate(i, data_matrix, labels_matrix, alphas, bias)

        if alpha_pairs_changed == 0:
            iteration += 1
        else:
            iteration = 0

    return alphas, bias