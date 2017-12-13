"""binary tree with linear regression applied to each leaf node"""
import numpy as np

def leaf_regression_weights(node_dataset):
    """calculate the regression weights for the instances of the current leaf node

    :type node_dataset: pandas.DataFrame
    :return: matrix of weights
    """
    X = np.mat(node_dataset.iloc[:, :-1].values)
    # add offset
    X = np.concatenate((np.mat(np.ones((X.shape[0], 1))), X), axis=1)
    y = np.mat(node_dataset.iloc[:, -1:].values)
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the min_instances value')
    weights = xTx.I * (X.T * y)
    return weights

def model_error(node_dataset):
    """calculates the residual sum of squares error of a leaf node of a model tree

    :type node_dataset: pandas.DataFrame
    :return: residual sum of squares error when applying linear regression to node_dataset
    """
    X = np.mat(node_dataset.iloc[:, :-1].values)
    # add offset
    X = np.concatenate((np.mat(np.ones((X.shape[0], 1))), X), axis=1)
    y = np.mat(node_dataset.iloc[:, -1:].values)
    regression_weights = leaf_regression_weights(node_dataset)
    y_hat = X * regression_weights
    rss = np.sum(np.power(y - y_hat, 2))
    return rss

def binary_split(dataset, feature, value):
    """splits a dataset into 2 subsets according to the value of a certain feature

    :type dataset: pandas.DataFrame
    :param dataset: data we want to split into subsets
    :param feature: feature used as criteria for the split
    :param value: value of the input feature used as threshold for the split
    :return: 2 subsets of the dataset as pandas.DataFrame
    """
    subset_left = dataset.loc[dataset[feature] <= value]
    subset_right = dataset.loc[dataset[feature] > value]
    return subset_left, subset_right

def choose_best_split(dataset, min_error_delta=1, min_instances=4):
    """finds the feature and value that splits the given dataset with the lowest error

    :type dataset: pandas.DataFrame
    :param dataset: data on which we want to find the optimal split feature and value
    :param min_error_delta: minimum decrease in error required for it to be a good split.
                            If decrease in error is lower than this value regression weights for leaf node are returned
    :param min_instances: minimum number of instances each subset must have.
                          If a subset has fewer number of instances regression weights for leaf node are returned
    :return: feature and value by which to make the optimal split
    """

    # find out best feature and value by which to make the split
    error_before_split = model_error(dataset)
    best_error_after_split = np.inf
    best_feature = dataset.columns[0]
    best_value = 0
    for feature in dataset.columns[:-1]: # loop only over independent variables
        for split_value in set(dataset.loc[:, feature].values):
            subset_left, subset_right = binary_split(dataset, feature, split_value)
            # ignore split if a subset does not have enough features
            if (subset_left.shape[0] < min_instances) or (subset_right.shape[0] < min_instances): continue
            error_after_split = model_error(subset_left) + model_error(subset_right)
            if error_after_split < best_error_after_split:
                best_error_after_split = error_after_split
                best_feature = feature
                best_value = split_value

    # if error decrease is not enough return regression weights for leaf node
    if (error_before_split - best_error_after_split) < min_error_delta:
        return None, leaf_regression_weights(dataset)

    # if resulting subsets are smaller than threshold return regression weights for leaf node
    subset_left, subset_right = binary_split(dataset, best_feature, best_value)
    if (subset_left.shape[0] < min_instances) or (subset_right.shape[0] < min_instances):
        return None, leaf_regression_weights(dataset)

    return best_feature, best_value