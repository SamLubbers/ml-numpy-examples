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