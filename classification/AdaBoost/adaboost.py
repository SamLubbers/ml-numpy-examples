from decision_stump import find_best_stump, classify_stump
import numpy as np

def adaboost_train_ds(data, labels, iterations=20):
    """adaboost algorithm to find the best decision stamps

    :param data: numpy.ndarray (m x n) of training set data
    :param labels: numpy.ndarray (m x 1) containing the labels associated to each instance in the training data
    :param iterations: number of iterations we want to run the adaboost algorithm
    :return: list of best decision stumps found over all iterations
    """
    all_stumps = []
    num_instances = data.shape[0]
    instance_weights = np.mat(np.ones((num_instances, 1))/num_instances)
    aggregate_prediction = np.mat(np.zeros((num_instances, 1)))
    for _ in range(iterations):
        best_stump, prediction, error = find_best_stump(data, labels, instance_weights)

        alpha = 0.5 * np.log((1.0-error)/max(error, 1e-16))
        best_stump['alpha'] = alpha
        all_stumps.append(best_stump)

        # correctly classified examples will decrease in weight and correctly classified will increase
        weight_update_sign = np.multiply(np.mat(labels), prediction) * -1
        instance_weights = np.multiply(instance_weights, np.exp(weight_update_sign * alpha)) / sum(instance_weights)

        aggregate_prediction += alpha * prediction

        error_rate = np.sum((np.sign(aggregate_prediction ) != labels).astype(int)) / num_instances

        if error_rate == 0: break

    return all_stumps

def adaboost_classify(data, labels, new_instances):
    """classifies a new instance using adaboost with multiple decision stumps

    :param data: numpy.ndarray (m x n) of training set data
    :param labels: numpy.ndarray (m x 1) containing the labels associated to each instance in the training data
    :param new_instance: numpy.ndarray (len(new_instaces) x n) of the new instance(s) we want to classify
    :return: predicted class of the new instance (-1, 1)
    """
    stumps = adaboost_train_ds(data, labels)
    num_new_instances = new_instances.shape[0]
    aggregate_prediction = np.mat(np.zeros((num_new_instances,1)))
    for stump in stumps:
        prediction = classify_stump(feature=np.mat(new_instances[:, stump['feature_index']]),
                                    split_value=stump['split_value'],
                                    class_assignment=stump['class_assignment'])

        aggregate_prediction += stump['alpha'] * prediction

    return np.sign(aggregate_prediction)
