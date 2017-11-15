import numpy as np
import operator


def classify_point(new_point, dataset, k):
    """
    labels an unlabelled instance of a dataset using the k nearest neighbours algorithm
    :type new_point: numpy.ndarray
    :param new_point: unlabelled instance we want to label
                      features must be numeric values in order to calculate the distance

    :type dataset: pandas.DataFrame
    :param dataset: pandas 2D dataframe containing labelled instances.
                    features must be numeric values in order to calculate the distance

    :type k: int
    :param k: number of nearest neighbours we examine

    :return: the label corresponding to new_point
    """
    # extract the data from the pandas dataframe
    points = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, len(dataset.columns) - 1].values

    # calculate euclidean distance from new point to all existing points
    num_instances = points.shape[0]
    diff = np.tile(new_point, (num_instances, 1)) - points
    squared_diff = diff ** 2
    sum_squared_diff = squared_diff.sum(axis=1)
    euclidean_distances = sum_squared_diff ** 0.5

    # count labels of k nearest neighbours
    points_sort_index = euclidean_distances.argsort()
    nearest_nieghbours_count = {}
    for i in range(k):
        label = labels[points_sort_index[i]]
        nearest_nieghbours_count[label] = nearest_nieghbours_count.get(label, 0) + 1

    # sort results to retrieve winning label
    sorted_class_count = sorted(nearest_nieghbours_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    winning_label = sorted_class_count[0][0]
    return winning_label

# example use of classify_point function

# create pandas dataframe
import pandas as pd
from collections import OrderedDict

raw_data = [OrderedDict({'x': 0, 'y': 0 , 'label': 'A'}),
            OrderedDict({'x': 0, 'y': 0.1 , 'label': 'A'}),
            OrderedDict({'x': 1, 'y': 1 , 'label': 'B'}),
            OrderedDict({'x': 1, 'y': 1.1 , 'label': 'B'})]

dataset = pd.DataFrame(raw_data)

# define new unlabelled instance
new = np.array([[2, 2]])

# classify unlabelled instance and display result
new_label = classify_point(new, dataset, 3)

print("point %s has label %s" % (str(new), new_label))