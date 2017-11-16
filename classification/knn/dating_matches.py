import pandas as pd
import numpy as np

# import dataset
def load_dataset():
    dataset = pd.read_csv('datingTestSet.txt', delimiter = '\t', header = None)
    
    dataset.columns = ["frequentflyer_miles",
                       "videogame_hours",
                       "icecream_liters",
                       "likes"]    
    return dataset

# data preprocessing
def encode_categorical(categorical_feature):
    """
    Encode labels of one categorical feature with value between 0 and n_categories-1.
    :type categorical_feature: numpy.ndarray
    :param categorical_feature: feature contanining different categories
    :return: numpy.ndarray with categories replaced by integers
    """
    categories = list(np.unique(categorical_feature))
    numerical_feature = np.zeros(categorical_feature.shape)
    num_instances = categorical_feature.shape[0]
    for i in range(num_instances):
        num_label = categories.index(categorical_feature[i])
        numerical_feature[i] = num_label
    
    return numerical_feature

def feature_scaling(features):
    """
    Rescaling is applied to the range of features to scale the range in [0, 1]
    :type features: numpy.ndarray
    :param features: unscalled set of features
    :return: scalled features with values in the range [0, 1]
    """
    min_vals = features.min(0)
    max_vals = features.max(0)
    ranges = max_vals - min_vals
    num_instances = features.shape[0]
    # rescaling operation
    scaled_features = features - np.tile(min_vals, (num_instances, 1))
    scaled_features = scaled_features / np.tile(ranges, (num_instances, 1))
    
    return scaled_features

# testing the performance of our knn algorithm on the data
from knn import classify_point

def dating_match_test():
    train_test_ratio = 0.10
    dataset = load_dataset()
    X = dataset.iloc[:, :-1].values
    X = feature_scaling(X)
    y = dataset.iloc[:, len(dataset.columns)-1].values
    y = encode_categorical(y)
    num_instances = X.shape[0]
    num_test_instances = int(num_instances*train_test_ratio)
    error_counter = 0.0
    for i in range(num_test_instances):
        predicted_label = classify_point(X[i, :], 
                                         X[num_test_instances:,:],
                                         y[num_test_instances:],
                                         3)

        if predicted_label != y[i] : error_counter += 1.0
    
    print("the total error rate is: %f" % (error_counter/float(num_test_instances)))

dating_match_test()