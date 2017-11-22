from math import log
import operator

def calculate_entropy(dataset):
    """calculates the entropy of the given dataset
    :param dataset: pandas.DataFrame
    :return: entropy of the given dataset
    """
    target_variable = dataset.iloc[:, len(dataset.columns) - 1].values
    label_count = {}
    for value in target_variable:
        label_count[value] = label_count.get(value, 0) + 1
    
    num_instances = len(target_variable)
    
    entropy = 0.0
    for count in label_count.values():
        prob = count/num_instances
        entropy -= prob * log(prob, 2)
    
    return entropy

def split_dataset(dataset, feature):
    """splits the dataset into subsets according to the specified feature.

    the number of subsets created equals the number of unique values for the given feature,
    with each subset being a collection of instances that share same value for the given feature
    :param dataset: pandas.DataFrame representing the dataset
    :param feature: name of the feature by which we want to split the dataset
    :return: list of subsets of the datasets, each subset of type pandas.DataFrame
    """ 
    subsets = []
    feature_unique_values= dataset[feature].unique()
    for value in feature_unique_values:
        subsets.append(dataset.loc[dataset[feature] == value])
        
    return subsets

def best_split_feature(dataset):
    """identifies the best feature by which to split a dataset to organise data in a decision tree

    the best feature is chosen according to the ID3 algorithm,
    which states that the best feature is the one with the maximum information gain
    :param dataset: pandas.DataFrame representing the dataset
    :return: feature by which to split the dataset, which will be the feature with the highest information gain
    """
    base_entropy = calculate_entropy(dataset)
    max_information_gain = 0.0
    best_feature = ''

    total_instances = dataset.shape[0]
    iv_subdataset = dataset.iloc[:, :-1] # subset of dataset with only independent variables
    for feature in iv_subdataset:
        subdatasets = split_dataset(dataset, feature)

        new_entropy = 0.0
        for subdataset in subdatasets:
            subdataset_instances = subdataset.shape[0]
            prob = subdataset_instances/total_instances
            new_entropy += (prob * calculate_entropy(subdataset))

        information_gain = base_entropy - new_entropy
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = feature

    return best_feature

def dominant_feature_value(feature):
    """returns the most dominant value of a given feature.

    used to assign a value to a leaf node with more than one label
    :type feature: pandas.series
    :param feature: feature of a dataset
    :return: dominant value of the given feature
    """
    unique_values = feature.unique()
    if len(unique_values) == 1: # return if there is feature only has one label
        return feature[0]

    value_count = {}
    for value in unique_values:
        value_count[value] = len(feature[feature == value])

    sorted_unique_values = sorted(value_count.items(),
                                  key=operator.itemgetter(1),
                                  reverse = True)
    dominant_value = sorted_unique_values[0][0]
    return dominant_value

def create_tree(dataset):
    """creates a decision tree out of the given dataset

    :type dataset: pandas.dataframe
    :type dataset: dataset of which we want to create the tree
    :return: nested dictionary representing the decision tree
    """
    num_features = len(dataset.columns)
    target_variable_vector = dataset.iloc[:, num_features - 1]
    unique_labels = target_variable_vector.unique()
    if len(unique_labels) == 1: # target variable values are all the same
        return unique_labels[0]
    if num_features == 2: # dataset only has one attribute and cannot be further split
        return dominant_feature_value(target_variable_vector)

    best_feature = best_split_feature(dataset)
    my_tree = {best_feature: {}}
    subsets = split_dataset(dataset, feature=best_feature)

    best_feature_unique_values = dataset[best_feature].unique()
    
    for value, subset in zip(best_feature_unique_values, subsets):
        my_tree[best_feature][value] = create_tree(subset)

    return my_tree