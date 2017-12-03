"""plotting the data and svm separating hyperplane"""
import matplotlib.pyplot as plt
import numpy as np
from usage import dataset, w, bias

# scatter data points with different colour for each class
class_0_points = dataset.loc[dataset.iloc[:, 2] == -1].iloc[:, [0, 1]].values
class_1_points = dataset.loc[dataset.iloc[:, 2] == 1].iloc[:, [0, 1]].values
plt.scatter(class_0_points[:, 0], class_0_points[:, 1], color='red')
plt.scatter(class_1_points[:, 0], class_1_points[:, 1], color='blue')

# plot hyperplane
hyperplane_x_points = np.arange(2, 6, 0.1)
# equal hyperplane equation to 0 and solve for y
hyperplane_y_points = (-float(w[0]) * hyperplane_x_points - float(bias)) / float(w[1])
plt.plot(hyperplane_x_points, hyperplane_y_points, color='green')
plt.show()
