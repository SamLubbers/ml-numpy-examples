from math import log

def calculate_feature_entropy(feature):
    label_count = {}
    for value in feature:
        label_count[value] = label_count.get(value, 0) + 1
    
    num_instances = len(feature)
    
    entropy = 0.0
    for count in label_count.values():
        prob = count/num_instances
        entropy -= prob * log(prob, 2)
    
    return entropy