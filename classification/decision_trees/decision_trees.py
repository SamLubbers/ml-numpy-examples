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