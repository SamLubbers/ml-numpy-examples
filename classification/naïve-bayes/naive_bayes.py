"""implementation of naive bayes algorithm"""
def calculate_priors(target_variable):
    """
    calculates the prior probability of each class contained in a vector
    :param target_variable: vector containing the different classes of the target_variable
    :return: dictionary with key values pairs of classes and their priors
    """
    classes, counts = np.unique(target_variable, return_counts=True)
    priors = np.round(counts/len(target_variable), decimals=4)

    return dict(zip(classes, priors))