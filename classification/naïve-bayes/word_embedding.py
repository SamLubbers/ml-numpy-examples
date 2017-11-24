"""convert text of words to a vector of features"""
import pandas as pd
from collections import OrderedDict

# data on restaurant reviews
raw_data = [OrderedDict({"review":"the food was incredible","positive":1}),
            OrderedDict({"review":"we had such a great time","positive":1}),
            OrderedDict({"review":"we loved the delicious starters they had","positive":1}),
            OrderedDict({"review":"the service was awful","positive":0}),
            OrderedDict({"review":"did not like the desserts","positive":0}),
            OrderedDict({"review":"the cutlery was not clean","positive":0}),
            ]

dataset = pd.DataFrame(raw_data)

def create_vocabulary(documents):
    """creates a list of all the words contained in a set of documents
    :type documents: numpy.ndarray or list of strings
    :param documents: documents can be a set of reviews, comments, tweets...
    :return: list of the individual words that appear in the documents
    """
    vocabulary_set = set([])
    for doc in documents:
        doc = doc.split()
        vocabulary_set = vocabulary_set | set(doc) # union of 2 sets
    
    return list(vocabulary_set)

vocabulary = create_vocabulary(dataset['review'].values)

def word2vector(vocab_list, text):
    """converts a text onto a vector with vocabulary words as features
    with value 1 if word occurs in text and 0 if it doesn't
    :param vocab_list: set of words in our vocabulary
    :param text: text we want to convert to a vector
    :return: vector representing text
    """
    word_vec = []
    text = text.split()
    for word in vocab_list:
        if word in text:
            word_vec.append(1)
        else:
            word_vec.append(0)
    return word_vec
