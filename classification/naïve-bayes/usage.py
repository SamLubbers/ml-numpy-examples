"""usage example of different functions used in naive bayes classfication"""
import pandas as pd
from collections import OrderedDict

# create data on restaurant reviews
raw_data = [OrderedDict({"review":"the food was good","positive":1}),
            OrderedDict({"review":"we had such a good time","positive":1}),
            OrderedDict({"review":"we loved the delicious starters they had","positive":1}),
            OrderedDict({"review":"the service was bad","positive":0}),
            OrderedDict({"review":"did not like the desserts","positive":0}),
            OrderedDict({"review":"the cutlery was bad","positive":0}),
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

# determine the label of a new category
from naive_bayes import classify_NB
new_review = "we had a very good time"
new_vector = word2vector(vocabulary, new_review)
new_review_label = classify_NB(new_vector, word_vectors, y)

if new_review_label == 1:
    print('the review \'%s\' is classified as positive' % new_review)
elif new_review_label == 0:
    print('the review \'%s\' is labelled as negative' % new_review)
    
