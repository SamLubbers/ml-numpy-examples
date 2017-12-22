"""singular value decomposition"""
import numpy as np

def singular_values(dataset):
    """applies svd to the dataset to obtain singular values

    :type dataset: numpy.array (mxn) | numpy.matrix (mxn)
    :return: singular values obtained from applying svd on the dataset
    """
    U, sigma, Vt = np.linalg.svd(dataset)
    return sigma

def compress_dataset(dataset):
    """uses svd to compress the dataset into a new one with fewer feature but that still conserves 90% of the energy

    :type dataset: numpy.matrix (mxn)
    :return: compressed dataset
    """
    U, sigma, Vt = np.linalg.svd(dataset)
    energy_threshold = sum(sigma ** 2) * 0.9

    # find minimum number of features that contain 90% of the energy
    num_single_values = 0
    for i in range(1, len(sigma)):
        energy = sum(sigma[:i] ** 2)
        if energy >= energy_threshold:
            num_single_values = i

    # reconstruct the dataset with less features
    sigma_matrix = np.mat(np.eye(num_single_values) * sigma[:num_single_values])
    # transform dataset into lower dimensional space
    compressed_dataset = dataset.T * U[:, :num_single_values] * sigma_matrix.I
    return compressed_dataset
