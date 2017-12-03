"""suport vector machine classifier"""

def classify_svm_linear(instance_matrix, w, bias):
    """classifies a new instance using the parameters that define the linear separating hyperplane

    :type instance_matrix: numpy.matrix (1, n)
    :param instance_matrix: instance we want to classify
    :type w: numpy.matrix (n, 1)
    :param w: vector of constants of our hyperplane equation
    :type bias: numpy.matrix (1, 1)
    :param bias: bias of our hyperplane equation
    :return: 1 if label is positive, -1 if label is negative
    """
    label = float(instance_matrix * w + bias)
    return 1 if label > 0 else -1