from usage import dataset, weights
import matplotlib.pyplot as plt
import numpy as np

# separate instnaces by labels
class_0_points = dataset.loc[dataset['label'] == 0].iloc[:, [1, 2]].values
class_1_points = dataset.loc[dataset['label'] == 1].iloc[:, [1, 2]].values

# plot data points
plt.scatter(class_0_points[:, 0], class_0_points[:, 1], c='red', marker='s')
plt.scatter(class_1_points[:, 0], class_1_points[:, 1], c='blue')

# plot decision boundary
x = np.arange(min(dataset.iloc[:, 1].values), max(dataset.iloc[:, 1].values), 0.1)
# we equal input to sigmoid function to 0 and solve for x2
x2 = (-weights[0]-weights[1]*x)/weights[2]
plt.plot(x, x2)
plt.show()
