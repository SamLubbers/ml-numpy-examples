"""example use of feature extraction with PCA"""
import numpy as np
import pandas as pd

dataset = np.mat(pd.read_csv('testSet.txt', delimiter='\t', header=None).values)

from pca import pca
new_dataset = pca(dataset)