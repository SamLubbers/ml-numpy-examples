"""example use of feature extraction with PCA"""
import numpy as np
import pandas as pd

dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None).values

from pca import pca
new_dataset, reconstructed_dataset = pca(np.mat(dataset), n_features=1)
