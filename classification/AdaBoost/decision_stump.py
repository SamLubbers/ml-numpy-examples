"""weak learner decision stump to be used in ensemble techniques like bagging or boosting"""
import numpy as np

def classify_stump(feature_values, split_value, class_assignment):
    """classifies all the instances according to a split value on a given feature

    all values above the split value will get one label and values below it a different one

    :param feature_values: numpy.matrix (m x 1) containing all values for a given feature
    :param split_value: value used for the classification of each instance
    :param class_assignment: specifies which side of the split gets which label
    :return: vector of predicted labels for all instances
    """
    prediction = np.ones(len(feature_values), 1)
    if class_assignment == 'less_than':
        prediction[feature_values <= split_value] = -1.0
    elif class_assignment == 'greater_than':
        prediction[feature_values > split_value] = -1.0

    return prediction