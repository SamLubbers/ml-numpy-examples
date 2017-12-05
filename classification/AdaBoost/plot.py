from usage import dataset, stumps
import matplotlib.pyplot as plt

# scatter datapoints
c1 = dataset.loc[dataset['label'] == -1].values
c2 = dataset.loc[dataset['label'] == 1].values

plt.scatter(c1[:, 0], c1[:, 1], color='red')
plt.scatter(c2[:, 0], c2[:, 1], color='blue')

# scatter decision stumps
import numpy as np
x_range = {'min': dataset.iloc[:, 0].min()-0.1, 'max': dataset.iloc[:, 0].max()+0.1}
y_range = {'min': dataset.iloc[:, 1].min()-0.1, 'max': dataset.iloc[:, 1].max()+0.1}

for stump in stumps:
    dimension = stump['feature_index']
    split_value = stump['split_value']

    if dimension == 0:
        y = np.arange(y_range['min'], y_range['max'], 0.1)
        x = np.tile(split_value, len(y))
    elif dimension == 1:
        x = np.arange(x_range['min'], x_range['max'], 0.1)
        y = np.tile(split_value, len(x))

    plt.plot(x, y, color='pink')

plt.show()
