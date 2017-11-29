import numpy as np

def sigmoid(z):
    """applies sigmoid function on z
    :param z: input to the sigmoid function
    :return: number between 0 and 1
    """
    return 1.0/(1.0+np.exp(-z))
