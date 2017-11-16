import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('datingTestSet.txt', delimiter = '\t', header = None)

dataset.columns = ["frequentflyer_miles",
                   "videogame_hours",
                   "icecream_liters",
                   "likes"]

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values

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
    
y = encode_categorical(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

