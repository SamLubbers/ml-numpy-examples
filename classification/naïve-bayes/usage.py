"""usage example of different functions used in naive bayes classfication"""
import pandas as pd
from collections import OrderedDict

# create data on restaurant reviews
raw_data = [OrderedDict({"review":"the food was incredible","positive":1}),
            OrderedDict({"review":"we had such a great time","positive":1}),
            OrderedDict({"review":"we loved the delicious starters they had","positive":1}),
            OrderedDict({"review":"the service was awful","positive":0}),
            OrderedDict({"review":"did not like the desserts","positive":0}),
            OrderedDict({"review":"the cutlery was not clean","positive":0}),
            ]

# convert data to pandas dataframe
dataset = pd.DataFrame(raw_data)
X = dataset['review'].values
y = dataset['positive'].values

from word_embedding import create_vocabulary, word2vector

vocabulary = create_vocabulary(X)

word_vectors = [word2vector(vocabulary, review) for review in X]

# calcualte probabilities
from naive_bayes import calculate_priors, calcualte_feature_likelihods
priors = calculate_priors(y)

feature_likelihoods = calcualte_feature_likelihods(word_vectors, y)

