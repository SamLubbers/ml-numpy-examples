from usage import dataset
import matplotlib.pyplot as plt

c1 = dataset.loc[dataset['label'] == -1].values
c2 = dataset.loc[dataset['label'] == 1].values

plt.scatter(c1[:, 0], c1[:, 1], color='red')
plt.scatter(c2[:, 0], c2[:, 1], color='blue')
plt.show()