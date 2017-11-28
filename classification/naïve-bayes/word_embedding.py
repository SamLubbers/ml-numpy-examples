"""convert text of words to a vector of features"""
import numpy as np
def create_vocabulary(documents):
    """creates a list of all the words contained in a set of documents
    :type documents: numpy.ndarray or list of strings
    :param documents: documents can be a set of reviews, comments, tweets...
    :return: list of the individual words that appear in the documents
    """
    vocabulary_set = set([])
    for doc in documents:
        vocabulary_set = vocabulary_set | set(doc) # union of 2 sets
    
    return list(vocabulary_set)


def word2vector(vocab_list, bag_of_words):
    """converts a bag of words onto a vector with vocabulary words as features
    with value 1 if word occurs in text and 0 if it doesn't
    :param vocab_list: set of words in our vocabulary
    :param bag_of_words: text we want to convert to a vector
    :return: vector representing text
    """
    word_vec = np.zeros(len(vocab_list))
    for word in bag_of_words:
        if word in vocab_list:
            word_vec[vocab_list.index(word)] += 1
    return word_vec
