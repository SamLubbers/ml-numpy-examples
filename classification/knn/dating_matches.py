import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('datingTestSet.txt', delimiter = '\t', header = None)

dataset.columns = ["frequentflyer_miles",
                   "videogame_hours",
                   "icecream_liters",
                   "likes"]

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values
