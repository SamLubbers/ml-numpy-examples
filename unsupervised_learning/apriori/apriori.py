"""apriori algorithm for association analysis"""
import numpy as np

def create_initial_itemsets(dataset):
    """creates all possible candidate itemsets of size 1 from the different transactions in our dataset

    :param dataset: list of transactions, each transaction being a list with different items
    :return: set containing candidate itemsets of size 1
    """
    initial_itemsets = [item for transaction in dataset for item in transaction]
    initial_itemsets.sort()
    initial_itemsets = set(initial_itemsets)
    initial_itemsets = [[item] for item in initial_itemsets]
    return list(map(frozenset, initial_itemsets)) # map to frozen set so we can use itemsets as keys for dicts

def filter_itemsets(dataset, itemsets, min_support):
    """filters our from itemsets each itemset with a support lower than min_support"""
    itemset_count = {}
    dataset = list(map(set, dataset)) # convert each transaction to set to eliminate duplicates
    for transaction in dataset:
            for itemset in itemsets:
            if itemset.issubset(transaction):
                itemset_count[itemset] = itemset_count.setdefault(itemset, 0) + 1

    num_items = len(dataset)
    filtered_itemsets = []
    itemests_support = {}
    for itemset, count in itemset_count.items():
        support = count / num_items
        if support >= min_support:
            filtered_itemsets.append(itemset)
        itemests_support[itemset] = support
    return filtered_itemsets, itemests_support
