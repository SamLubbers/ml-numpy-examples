"""use the decision tree to classify a new instance"""
import pandas as pd
from collections import OrderedDict

# create sample dataset
data_dict = [OrderedDict({"v1": 1, "v2": 1, "label":'yes'}),
             OrderedDict({"v1": 1, "v2": 1, "label":'yes'}),
             OrderedDict({"v1": 1, "v2": 0, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 1, "label":'no'}),
             OrderedDict({"v1": 0, "v2": 1, "label": 'no'})]
dataset = pd.DataFrame(data_dict)

# create decision tree
from decision_trees import create_tree
my_tree = create_tree(dataset)


# create new instance
new = pd.Series(OrderedDict({"v1": 1, "v2": 1}))

# make new prediction using decision tree
def classify(decision_tree, new_instance):
    node_feature = next(iter(decision_tree.keys()))
    node_subnodes = decision_tree[node_feature]
    for node_value in node_subnodes.keys():
        if node_value == new_instance[node_feature]:
            if type(node_subnodes[node_value]) is dict:
                return classify(node_subnodes[node_value], new_instance)
            else:
                return node_subnodes[node_value]
    
label = classify(my_tree, new)
print('the predicted label for is: %s' % label)