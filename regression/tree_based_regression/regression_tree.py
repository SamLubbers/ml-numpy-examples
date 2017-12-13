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
    for feature in dataset.columns[:-1]: # loop only over independent variables
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
    my_tree['left'] = create_tree(subset_left, min_error_delta, min_instances)
    my_tree['right'] = create_tree(subset_right, min_error_delta, min_instances)
    return my_tree

def is_tree(obj):
    """assesses weather the current object is a tree or leaf node"""
    if type(obj) is dict:
        return True
    return False

def tree_mean_value(my_tree):
    """calculates the mean value of a tree
    :type my_tree: dict
    """
    my_tree = my_tree.copy()
    if is_tree(my_tree['left']): my_tree['left'] = tree_mean_value(my_tree['left'])
    if is_tree(my_tree['right']): my_tree['right'] = tree_mean_value(my_tree['right'])
    return (my_tree['left'] + my_tree['right']) / 2

def prune(my_tree, test_data):
    """postpruning applied recursively to my_tree

    prune is called recursively on each subnode that is a tree
    when both subnodes are leaf nodes it merges them if the merged nodes have a lower error rate on the test_data
    :type my_tree: dict
    :param my_tree: regression tree
    :type test_data: pandas.DataFrame
    :param test_data: test data used to decide whether 2 leave nodes should be merged
    :return: pruned tree
    """
    if test_data.shape[0] == 0: return tree_mean_value(my_tree) # collapse tree if test data is empty

    test_subset_left, test_subset_right = binary_split(test_data, my_tree['split_feature'], my_tree['split_value'])

    if is_tree(my_tree['left']): my_tree['left'] = prune(my_tree['left'], test_subset_left)
    if is_tree(my_tree['right']): my_tree['right'] = prune(my_tree['right'], test_subset_right)
    # if both nodes are leaf nodes calculate errors and decide whether to merge leafs
    if not is_tree(my_tree['left']) and not is_tree(my_tree['right']):
        error_no_merge = np.sum(np.power(test_subset_left.iloc[:, -1] - my_tree['left'], 2)) + \
                         np.sum(np.power(test_subset_right.iloc[:, -1] - my_tree['right'], 2))

        tree_mean = tree_mean_value(my_tree)
        error_merge = np.sum(np.power(test_data.iloc[:, -1] - tree_mean, 2))
        # merge leaf nodes if they give a better prediction
        if error_merge < error_no_merge:
            return tree_mean
    return my_tree
