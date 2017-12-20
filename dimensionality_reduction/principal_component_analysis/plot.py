"""plot of original data and first principal component (maximum variance)"""
from usage import dataset, reconstructed_dataset
import matplotlib.pyplot as plt

plt.scatter(dataset[:,0], dataset[:,1])
plt.scatter(reconstructed_dataset[:,0], reconstructed_dataset[:,1], color='red')

plt.show()
