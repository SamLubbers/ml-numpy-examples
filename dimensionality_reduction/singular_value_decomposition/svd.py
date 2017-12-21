"""singular value decomposition"""
import numpy as np

def singular_values(dataset):
    """applies svd to the dataset to obtain singular values

    :type dataset: numpy.array (mxn)
    :return: singular values obtained from applying svd on the dataset
    """
    U, sigma, Vt = np.linalg.svd(dataset)
    return sigma
