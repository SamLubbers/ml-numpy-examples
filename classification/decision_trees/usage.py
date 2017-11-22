"""
example usage of the decision_tree module
"""

# create dataset
from decision_trees import calculate_entropy, split_dataset
import pandas as pd
from collections import OrderedDict

data_dict = [OrderedDict({"v1": 1, "v2": 1, "label":'yes'}),
             OrderedDict({"v1": 1, "v2": 0, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 1, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 0, "label":'no'})]
dataset = pd.DataFrame(data_dict)

# calculate the entropy for our target variable
entropy = calculate_entropy(dataset)
print('entropy of the dataset is %f\n' % entropy)

# split dataset according to the first feature
split_feature = 'v1'
subsets = split_dataset(dataset, split_feature)
print('the subsets of splitting the dataset by feature %s are:' % split_feature)
for subset in subsets:
    print(subset)