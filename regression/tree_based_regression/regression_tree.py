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
    subset_left = dataset.loc[dataset[feature] <= value]
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

def choose_best_split(dataset, min_error_delta=1, min_instances=4):
    """finds the feature and value that splits the given dataset with the lowest error

    :type dataset: pandas.DataFrame
    :param dataset: data on which we want to find the optimal split feature and value
    :param min_error_delta: minimum decrease in error required for it to be a good split.
                            If decrease in error is lower than this value a leaf node is returned
    :param min_instances: minimum number of instances each subset must have.
                          If a subset has fewer number of instances a leaf node is returned
    :return: feature and value by which to make the optimal split
    """
    # return leaf node if all values of the target variable are equal
    if len(set(dataset.iloc[:, -1].values.tolist())) == 1:
        return None, calculate_leaf_value(dataset)

    # find out best feature and value by which to make the split
    error_before_split = calculate_error(dataset)
    best_error_after_split = np.inf
    best_feature = dataset.columns[0]
    best_value = 0
    for feature in dataset.columns:
        for split_value in set(dataset.loc[:, feature].values):
            subset_left, subset_right = binary_split(dataset, feature, split_value)
            # ignore split if a subset does not have enough features
            if (subset_left.shape[0] < min_instances) or (subset_right.shape[0] < min_instances): continue
            error_after_split = calculate_error(subset_left) + calculate_error(subset_right)
            if error_after_split < best_error_after_split:
                best_error_after_split = error_after_split
                best_feature = feature
                best_value = split_value

    # if error decrease is not enough return leaf node
    if (error_before_split - best_error_after_split) < min_error_delta:
        return None, calculate_leaf_value(dataset)

    # if resulting subsets are smaller than threshold return leaf node
    subset_left, subset_right = binary_split(dataset, best_feature, best_value)
    if (subset_left.shape[0] < min_instances) or (subset_right.shape[0] < min_instances):
        return None, calculate_leaf_value(dataset)

    return best_feature, best_value

def create_tree(dataset, min_error_delta=1, min_instances=4):
    """create a regression tree for the given dataset

    :type dataset: pandas.DataFrame
    :param dataset: data on which we want to create the decision tree
    :param min_error_delta: minimum decrease in error required for it to be a good split.
    :param min_instances: minimum number of instances each subset must have.
    :return: dictionary representing the regression tree
    """
    split_feature, split_value = choose_best_split(dataset, min_error_delta, min_instances)
    if split_feature is None:
        return split_value
    subset_left, subset_right = binary_split(dataset, split_feature, split_value)

    my_tree = {}
    my_tree['split_feature'] = split_feature
    my_tree['split_value'] = split_value
    my_tree['subset_left'] = create_tree(subset_left, min_error_delta, min_instances)
    my_tree['subset_right'] = create_tree(subset_right, min_error_delta, min_instances)
    return my_tree