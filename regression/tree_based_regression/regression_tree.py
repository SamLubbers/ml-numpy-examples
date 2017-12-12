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


def calculate_leaf_value(dataset):
    """calculates the average value of the target variable of the given dataset"""
    return np.mean(dataset.iloc[:, -1].values)

def calculate_error(dataset):
    """calculates the total squared error of the given dataset"""
    mean_square_error = np.var(dataset.iloc[:, -1].values)
    num_instances = dataset.values.shape[0]
    total_square_error = mean_square_error * num_instances
    return total_square_error
    
# TODO function that decides best split feature

# TODO function that builds tree