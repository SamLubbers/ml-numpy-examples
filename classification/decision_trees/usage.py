"""
example usage of the decision_tree module
"""

# create dataset
from decision_trees import calculate_entropy
import pandas as pd
from collections import OrderedDict

data_dict = [OrderedDict({"v1": 1, "v2": 1, "label":'yes'}),
             OrderedDict({"v1": 1, "v2": 0, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 1, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 0, "label":'no'})]
dataset = pd.DataFrame(data_dict)

# calculate the entropy for our target variable
entropy = calculate_entropy(dataset)
print('entropy of the dataset is %f' % entropy)