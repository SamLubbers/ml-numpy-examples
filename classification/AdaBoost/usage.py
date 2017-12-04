from collections import OrderedDict
import pandas as pd

raw_data = [OrderedDict({'x1': 1.0, 'x2': 2.1 , 'label': 1}),
            OrderedDict({'x1': 1.3, 'x2': 1.0 , 'label': -1}),
            OrderedDict({'x1': 2.0, 'x2': 1.0 , 'label': 1}),
            OrderedDict({'x1': 1.0, 'x2': 1.0 , 'label': -1}),
            OrderedDict({'x1': 1.5, 'x2': 1.6, 'label': 1})]

dataset = pd.DataFrame(raw_data)

# create numpy.ndarray of features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from decision_stump import find_best_stump
import numpy as np
weights = np.mat(np.ones((len(y), 1))/5)
best_stump, best_prediction, min_error = find_best_stump(X, y, weights))
