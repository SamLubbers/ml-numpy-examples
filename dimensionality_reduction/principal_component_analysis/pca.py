"""principal component analysis for feature extraction"""
import numpy as np

def pca(dataset, n_features=3):
    """creates new dataset with fewer features using principal component analysis feature extraction

    it will return a dataset with n_features number of features or as many as the original dataset has
    :type dataset: numpy.matrix (mxn)
    :param n_features: number of principal components (number of features of the new dataset)
    :return: dataset with best extracted features, and reconstructed dataset using those features
    """
    mean = np.mean(dataset, axis=0)
    dataset_distance_mean = dataset - mean
    covariance_matrix = np.cov(dataset_distance_mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(np.mat(covariance_matrix))
    # get eigenvectors with largest eigenvalues
    eigenvalues_indexes = np.argsort(eigenvalues)
    eigenvalues_indexes = eigenvalues_indexes[: -(n_features + 1): -1]
    top_eigenvectors = eigenvectors[:, eigenvalues_indexes]
    # use these eigenvectors to create new dataset with principal components
    new_dataset = dataset_distance_mean * top_eigenvectors
    reconstructed_dataset = (new_dataset * top_eigenvectors.T) + mean
    # reconstruct dataset using principal component
    return np.array(new_dataset), np.array(reconstructed_dataset)

def principal_component_variance(dataset):
    """calculates the percentage of variance each principal component explains

    :type dataset: numpy.matrix (mxn)
    :param n_features: number of features of the new dataset
    :return: list containing the percentage of variance each principal component explains
    """
    mean = np.mean(dataset, axis=0)
    dataset_distance_mean = dataset - mean
    covariance_matrix = np.cov(dataset_distance_mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(np.mat(covariance_matrix))
    variance_percentage = [((eigenvalue/sum(eigenvalues)) * 100) for eigenvalue in eigenvalues]
    return variance_percentage