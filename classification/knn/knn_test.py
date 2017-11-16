# example use of classify_point function

# create pandas dataframe
import pandas as pd
from collections import OrderedDict

raw_data = [OrderedDict({'x': 0, 'y': 0 , 'label': 'A'}),
            OrderedDict({'x': 0, 'y': 0.1 , 'label': 'A'}),
            OrderedDict({'x': 1, 'y': 1 , 'label': 'B'}),
            OrderedDict({'x': 1, 'y': 1.1 , 'label': 'B'})]

dataset = pd.DataFrame(raw_data)

# create numpy.ndarray of features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values

# define new unlabelled instance
import numpy as np
new = np.array([[2, 2]])

# project must be set as working directory for relative input to work
from classification.knn.knn import classify_point
# classify unlabelled instance and display result
new_label = classify_point(new, X, y, 3)

print("point %s has label %s" % (str(new), new_label))
