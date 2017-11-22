from math import log


def calculate_entropy(dataset):
    """
    calculates the entropy of the given dataset
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
    """
    splits the dataset into subsets according to the specified feature.
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
