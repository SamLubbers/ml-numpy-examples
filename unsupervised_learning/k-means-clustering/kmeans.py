"""kmeans clustering algorithm"""
import numpy as np

def random_centroid(dataset, k):
    """initializes random values for the centroids within the dataset boundaries

    :type dataset: numpy.ndarray (mxn)
    :param dataset: data on which we will perform the clustering
    :param k: number of clusters
    :return: matrix with centroids for each of the k clusters
    """
    num_instances = dataset.shape[1]
    centroids = np.mat(np.zeros((k, num_instances)))
    for feature_index, feature in enumerate(dataset.T):
        feature_min = np.min(feature)
        feature_max = np.max(feature)
        feature_range = feature_max - feature_min
        centroids[:, feature_index] = feature_min + feature_range * np.random.rand(k, 1)
    return centroids