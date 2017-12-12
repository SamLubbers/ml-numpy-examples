"""tree based regression algorithm"""
import numpy as np

def binary_split(dataset, feature, value):
    """splits a dataset into 2 subsets according to the value of a certain feature

    :type dataset: pandas.DataFrame
    :param dataset: data we want to split into subsets
    :param feature: feature used as criteria for the split
    :param value: value of the input feature used as threshold for the split
    :return: 2 subsets of the dataset as pandas.DataFrame
    """
    subset_left = dataset.loc[dataset[feature] < value]
    subset_right = dataset.loc[dataset[feature] > value]
    return subset_left, subset_right

# TODO function that decides best split feature

# TODO function that builds tree