"""
example usage of the decision_tree module
"""

# create dataset
from decision_trees import calculate_feature_entropy
import pandas as pd
from collections import OrderedDict

data_dict = [OrderedDict({"v1": 1, "v2": 1, "label":'yes'}),
             OrderedDict({"v1": 1, "v2": 0, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 1, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 0, "label":'no'})]
dataset = pd.DataFrame(data_dict)

# calculate the entropy for our target variable
target_variable = dataset.iloc[:, len(dataset.columns) - 1].values
target_variable_entropy = calculate_feature_entropy(target_variable)
print('entropy of the target variable is %f' % target_variable_entropy)