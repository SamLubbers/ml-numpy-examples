"""apriori algorithm for association analysis"""
import numpy as np

def create_initial_itemsets(dataset):
    """creates all possible candidate itemsets of size 1 from the different transactions in our dataset

    :param dataset: list of transactions, each transaction being a list with different items
    :return: set containing candidate itemsets of size 1
    """
    initial_itemsets = [item for transaction in dataset for item in transaction]
    initial_itemsets.sort()
    return set(initial_itemsets)