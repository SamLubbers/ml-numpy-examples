"""Sequential Minimal Optimization used to figure out the optimal separating hyperplane"""
import random
import numpy as np

def select_alpha2(alpha1, num_alphas):
    """a random value is chosen for alpha2 as long as it does not equal alpha1

    :param alpha1: value which alpha2 cannot take
    :param num_alphas: total number of alphas
    :return: integer that will be the value for alpha2
    """
    alpha2 = alpha1
    while(alpha2==alpha1):
        alpha2 = random.uniform(0, num_alphas)

    return alpha2

def bound_alpha(alpha, max_limit, min_limit):
    """bounds alpha value between certain limits"""
    if alpha > max_limit:
        alpha = max_limit
    if alpha < min_limit:
        alpha = min_limit

    return alpha

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
    labels_matrix = np.mat(labels).transpose()
    # initialize alphas and bias
    m, n = data_matrix.shape
    alphas = np.zeros((m, 1))
    bias = 0
    # run optimization
    iteration = 0
    while(iteration<max_iterations):
        alpha_pairs_changed = 0
        # TODO write optimization code here
        if alpha_pairs_changed == 0:
            iteration += 1
        else:
            iteration = 0
            
    return alphas, bias