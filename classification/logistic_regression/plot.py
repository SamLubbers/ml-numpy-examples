from usage import dataset, weights, weights_stochastic
import matplotlib.pyplot as plt
import numpy as np

# separate instnaces by labels
class_0_points = dataset.loc[dataset['label'] == 0].iloc[:, [1, 2]].values
class_1_points = dataset.loc[dataset['label'] == 1].iloc[:, [1, 2]].values

# plot data points
plt.scatter(class_0_points[:, 0], class_0_points[:, 1], c='red', marker='s')
plt.scatter(class_1_points[:, 0], class_1_points[:, 1], c='blue')

# plot decision boundary - gradient descent
x = np.arange(min(dataset.iloc[:, 1].values), max(dataset.iloc[:, 1].values), 0.1)
# we equal input to sigmoid function to 0 and solve for x2
x2 = (-weights[0]-weights[1]*x)/weights[2]
plt.plot(x, x2)

# plot decision boundary - stochastic gradient descent
x2_s = (-weights_stochastic[0]-weights_stochastic[1]*x)/weights_stochastic[2]
plt.plot(x, x2_s)

plt.show()