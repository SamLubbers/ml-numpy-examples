"""example use of functions in the kmeans module"""

# loading dataset
import pandas as pd
dataset = pd.read_csv('testSet.txt', delimiter='\t', header=None).values

# initialize random centroids on the dataset
from kmeans import random_centroids
rand_centroids = random_centroids(dataset, k=2)

# euclidean distance from point to centroid
from kmeans import euclidean_distance
distance = euclidean_distance(dataset[0,:], rand_centroids[0, :])

# kmeans clustering of dataset
from kmeans import kmeans
centroids, cluster_assignments = kmeans(dataset, k=4)