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
    """determines if a certain value of alpha can be optimized

    - the error rate for the current instance must surpass the tolerance
    - alphas must not be bounded
    """
    if ((label * error < -tolerance) and (alpha < C)) or ((label * error > tolerance) and (alpha > 0)):
        return True
    return False

def define_limits(label_i, label_j, alpha_i, alpha_j, C):
    """defines min and max limits given certain values of alpha"""
    if label_i != label_j:
        min_limit = max(0, alpha_j - alpha_i)
        max_limit = min(C, C + alpha_j - alpha_i)
    else:
        min_limit = max(0, alpha_j + alpha_i - C)
        max_limit = min(C, alpha_j + alpha_i)
    return min_limit, max_limit

def optimal_change_factor(instance_i, instance_j):
    """calculates the optimal amount by which to change alpha_j

    :param instance_i: matrix of features of instance i of the data: data_matrix[i, :]
    :param instance_j: matrix of features of instance i of the data: data_matrix[j, :]
    :return: optimal change factor of alpha_j
    """

    ocf = 2.0 * instance_i * instance_j.T - instance_i * instance_i.T - instance_j * instance_j.T
    return ocf


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
    min_alpha_change = 0.00001
    iteration = 0
    while(iteration<max_iterations):
        alpha_pairs_changed = 0
        for i in range(m):
            error_i = instance_error_rate(i, data_matrix, labels_matrix, alphas, bias)
            if can_be_optimized(alphas[i], labels_matrix[i], error_i, C, tolerance):

                # if alpha can be optimized get another instance at random and calculate its error rate
                j = random_index(i, m)
                error_j = instance_error_rate(j, data_matrix, labels_matrix, alphas, bias)

                #  make sure alpha stays between 0 and C
                min_limit, max_limit = define_limits(labels_matrix[i],
                                                     labels_matrix[j],
                                                     alphas[i],
                                                     alphas[j],
                                                     C)

                # if alpha values for this instance are bound move onto next instance
                if min_limit == max_limit: continue

                # calculate optimal amount by which to change alpha_j
                ocf = optimal_change_factor(instance_i=data_matrix[i, :], instance_j=data_matrix[j, :])

                # move onto next instance if optimal change factor is 0
                # this is an oversimplification of the original smo algorithm
                if ocf >= 0: continue

                # define old alpha values
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # calculate new alpha value of j
                alphas[j] -= labels_matrix[j] * (error_i - error_j) / ocf
                alphas[j] = bound_alpha(alphas[j], max_limit, min_limit)

                # if change in alpha is not notable move onto next instances
                if abs(alphas[j] - alpha_j_old) < min_alpha_change: continue

                # update alpha_i by the same amount as alpha_j in the opposite direction
                alphas[i] += labels_matrix[j] * labels_matrix[i] * (alpha_j_old - alphas[j])

                # obtain bias values for each alpha
                bias_i = bias \
                         - error_i \
                         - labels_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i,:] * data_matrix[i, :].T \
                         - labels_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T

                bias_j = bias \
                         - error_j \
                         - labels_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i,:] * data_matrix[j, :].T \
                         - labels_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T

                # define new bias
                if (0 < alphas[i]) and (alphas[i] < C): bias = bias_i
                elif (0 < alphas[j]) and (alphas[j] < C): bias = bias_j
                else: bias = (bias_i + bias_j) / 2.0

                # increment count
                alpha_pairs_changed += 1

        if alpha_pairs_changed == 0:
            iteration += 1
        else:
            iteration = 0

    return alphas, bias