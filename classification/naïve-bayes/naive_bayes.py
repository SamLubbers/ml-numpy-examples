"""implementation of naive bayes algorithm"""
import numpy as np

def calculate_priors(target_variable):
    """
    calculates the prior probability of each class contained in a vector
    :param target_variable: vector containing the different categories of the target_variable
    :return: dictionary with key values pairs of categories and their priors
    """
    categories, counts = np.unique(target_variable, return_counts=True)
    priors = np.round(counts/len(target_variable), decimals=4)

    return dict(zip(categories, priors))

def calcualte_feature_likelihods(vector_documents, target_variable):
    """calculates the likelihood of each feature of a document for each category

    for each category, it calculates the likelihood p(feature|category)
    by dividing the number of times a feature occurs for by the total number of occurring features

    This will result in an array representing of indicative each feature is of each category
    :param vector_documents: set of vectors representing words in a document
    :param target_variable: vector containing category associated to each instance
    :return: dictionary with the likelihood of each feature of a document for each category
    """
    feature_count = {} # counts how often each specific feature occurs for each class
    total_features = {} # counts the total amount of occuring features for each class
    categories = np.unique(target_variable)
    num_features = len(vector_documents[0])
    for category in categories:
        # we do not initialize feature counts with 0 to avoid probabilities of 0
        feature_count[category] = np.ones(num_features)
        total_features[category] = 2.0

    for index, category in enumerate(target_variable):
        feature_count[category] += vector_documents[index]
        total_features[category] += np.sum(vector_documents[index])

    feature_likelihoods = {}
    for category in categories:
        # apply natural logarithm to avoid underflow
        feature_likelihoods[category] = np.log(feature_count[category]/total_features[category])

    return feature_likelihoods