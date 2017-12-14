"""kmeans clustering algorithm"""
import numpy as np

def random_centroids(dataset, k):
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

def euclidean_distance(vec_a, vec_b):
    """calculates the euclidean distance between 2 vectors"""
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))

def kmeans(dataset, k, initialize_centroids=random_centroids, calculate_distance=euclidean_distance):
    """clustering of instances in dataset into k clusters

    :type dataset: numpy.ndarray (mxn)
    :param dataset: data on which we will perform the clustering
    :param k: number of clusters
    :param initialize_centroids: function used to initialize centroids
    :param calculate_distance: function used to calculate distance between centroids and points
    :return: matrix of cluster assigned to each instance along with its associated error
    """
    num_instances = dataset.shape[0]
    cluster_assignment = np.mat(np.zeros((num_instances, 2)))
    centroids = initialize_centroids(dataset, k=k)

    clusters_changed = True
    while clusters_changed:
        clusters_changed = False
        # assign each point to the cluster with the closest centroid
        for i, instance in enumerate(dataset):
            closest_cluster = -1
            min_distance = np.inf # minimum calculate_distance from instance to centroid
            for cluster, centroid in enumerate(centroids):
                distance = calculate_distance(instance, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster
            if cluster_assignment[i, 0] != closest_cluster: clusters_changed = True
            cluster_assignment[i, 0] = closest_cluster
            error = min_distance ** 2
            cluster_assignment[i, 1] = error
        # update centroids
        for cluster, _ in enumerate(centroids):
            cluster_instances = dataset[np.nonzero(cluster_assignment[:, 0].A == cluster)[0]]
            centroids[cluster, :] = np.mean(cluster_instances, axis=0)

    return centroids, cluster_assignment

def bisecting_kmeans(dataset, k, calculate_distance=euclidean_distance):
    """clustering of instances in dataset into k clusters using bisecting k means

    :type dataset: numpy.ndarray (mxn)
    :param dataset: data on which we will perform the clustering
    :param k: number of clusters
    :param calculate_distance: function used to calculate distance between centroids and points
    :return: matrix of cluster assigned to each instance along with its associated error
    """
    num_instances = dataset.shape[0]
    cluster_assignment = np.mat(np.zeros((num_instances, 2)))
    # start by assigning all instances to one cluster
    initial_centroid = np.mat(np.mean(dataset, axis=0))
    for i, instance in enumerate(dataset):
        error = calculate_distance(initial_centroid, instance) ** 2
        cluster_assignment[i, 1] = error

    centroids = [initial_centroid]
    while len(centroids) < k:
        lowest_error = np.inf
        for cluster, _ in enumerate(centroids):
            # calculate error of the instances not corresponding to this cluster
            error_other_clusters = np.sum(cluster_assignment[np.nonzero(cluster_assignment[:, 0].A != cluster)[0], 1])
            # divide current cluster in 2 and calculate error for subclusters
            instances_in_cluster = dataset[np.nonzero(cluster_assignment[:, 0].A == cluster)[0]]
            new_centroids, new_clusters = kmeans(instances_in_cluster, 2, calculate_distance=euclidean_distance)
            error_subclusters = np.sum(new_clusters[:, 1])
            total_error = error_other_clusters + error_subclusters
            if total_error < lowest_error:
                lowest_error = total_error
                best_cluster = cluster
                best_new_centroids = new_centroids.copy()
                best_new_clusters = new_clusters.copy()

        # update cluster assignment
        best_new_clusters[np.nonzero(best_new_clusters[:, 0].A == 0)[0], 0] = best_cluster
        best_new_clusters[np.nonzero(best_new_clusters[:, 0].A == 1)[0], 0] = len(centroids)
        cluster_assignment[np.nonzero(cluster_assignment[:, 0].A == best_cluster)[0], :] = best_new_clusters

        # update centroids
        centroids[best_cluster] = best_new_centroids[0, :]
        centroids.append(best_new_centroids[1, :])

    # convert list of centroids from list to matrix
    centroids = np.mat(np.asarray(centroids).reshape(len(centroids), centroids[0].shape[1]))
    return centroids, cluster_assignment
