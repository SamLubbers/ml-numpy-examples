"""weak learner decision stump to be used in ensemble techniques like bagging or boosting"""
import numpy as np

def classify_stump(feature, split_value, class_assignment):
    """classifies all the instances according to a split value on a given feature

    all values above the split value will get one label and values below it a different one

    :param feature: numpy.matrix (m x 1) containing all values for a given feature
    :param split_value: value used for the classification of each instance
    :param class_assignment: specifies which side of the split gets which label
    :return: vector of predicted labels for all instances
    """
    feature = feature.T
    prediction = np.ones((len(feature), 1))
    if class_assignment == 'less_than':
        prediction[feature <= split_value] = -1.0
    elif class_assignment == 'greater_than':
        prediction[feature > split_value] = -1.0

    return prediction

def find_best_stump(data, labels, instance_weights):
    """finds the stump with the lowest weighted_error, according to the instance_weights

    weighted_error is calculated as the sum of the instance weights of those instances that have been predicted incorrectly
    :param data: numpy.ndarray (m x n) of training set data
    :param labels: numpy.ndarray (m x 1) containing the labels associated to each instance in the training data
    :param instance_weights: numpy.matrix (m x 1) weight associated to each instance
    :return: best stump, best predicted labels, minimum weighted error
    """
    # convert to matrix for easier operations
    data_matrix = np.mat(data)
    labels_matrix = np.mat(labels)

    # return variables
    best_stump = {'feature_index': None,
                  'split_value': None}
    best_prediction = np.mat(np.zeros((len(labels), 1)))
    min_weighted_error = np.inf

    # algorithm
    num_steps = 10 # number of splits examined over the range of values of each feature
    for feature_index, feature in enumerate(data_matrix.T):
        step_size = (feature.max() - feature.min()) / num_steps
        for step_number in range(-1, (num_steps + 1)):
            # different classes assigned to each side of the split
            for class_assignment in ['less_than', 'greater_than']:
                split_value = feature.min() + (step_size * step_number)
                prediction = classify_stump(feature, split_value, class_assignment)
                prediction_errors = (prediction != labels_matrix).astype(int)
                weighted_error = instance_weights.T * prediction_errors

                if weighted_error < min_weighted_error:
                    min_weighted_error = weighted_error.item(0)
                    best_stump['feature_index'] = feature_index
                    best_stump['split_value'] = split_value
                    best_prediction = prediction.copy()

    return best_stump, best_prediction, min_weighted_error