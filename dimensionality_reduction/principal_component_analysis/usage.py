"""example use of feature extraction with PCA"""
import numpy as np
import pandas as pd

# apply pca of a simple dataset
dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None).values

from pca import pca
new_dataset, reconstructed_dataset = pca(np.mat(dataset), n_features=1)

dataset_many_feautes = pd.read_csv('secom.data', delimiter=' ', header=None)
dataset_many_feautes = dataset_many_feautes.fillna(dataset_many_feautes.mean()) # replace NaN values with mean

# calculate component variance on large dataset
from pca import principal_component_variance
component_variance_percentage = principal_component_variance(dataset_many_feautes)

# show how much cumulative variance is covered by i number of principal compoents
n_principal_components = 20
for i in range(1, n_principal_components+1):
    print('the cumulative variance with %d principal components is %f' % 
          (i, sum(component_variance_percentage[:i])))