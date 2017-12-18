"""apriori algorithm for association analysis"""

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

def itemset_combinations(itemsets):
    """creates combinations of itemsets of size: len(itemsets) +1

    :param itemsets: list of itemsets of same size (same number items in each itemset)
    """
    new_itemsets = []
    new_itemsets_length = len(itemsets[0]) + 1
    for a in itemsets:
        for b in itemsets:
            new_itemset = a | b
            if (a != b) and (len(new_itemset) == new_itemsets_length) and (new_itemset not in new_itemsets):
                new_itemsets.append(new_itemset)

    return new_itemsets


def apriori(dataset, min_support=0.5):
    """finds the most frequent imtemsets of a dataset using the apriori algorithm

    :param dataset: list of transactions, each transaction being a list with different items
    :param min_support: minimum support required for an itemset not to be discated
    :return: list of most frequent itemsets of the dataset of transactions
    """
    initial_itemsets = create_initial_itemsets(dataset)
    itemsets, support = filter_itemsets(dataset, initial_itemsets, min_support=min_support)
    all_itemsets = [itemsets]

    while itemsets:  # loop until no new itemsets are created
        new_itemsets = itemset_combinations(itemsets)
        itemsets, itemsets_support = filter_itemsets(dataset, new_itemsets, min_support=min_support)
        support.update(itemsets_support)
        all_itemsets.append(itemsets)

    # eliminate nested itemsets
    all_itemsets = [itemset for m in all_itemsets for itemset in m]
    return all_itemsets, support

def itemset_rules(itemeset, consequents, itemsets_support, minimum_confidence=0.7):
    """finds relevant associations rules of items in the itemset with consequents

    :param itemeset: set of items from which the associations rules will be created
    :param consequents: possible consequents in the assocation rule
    :param itemsets_support: support of all itemsets
    :param minimum_confidence: minimum confidence for an association rule to be relevant
    :return: list of rules, each being a dict containing the rule antecedent, consequent and confidence
    """
    rules = []
    for consequent in consequents:
        antecedent = itemeset - consequent
        confidence = itemsets_support[itemeset] / itemsets_support[antecedent]
        if confidence >= minimum_confidence:
            rule = {'antecedent': antecedent,
                           'consequent': consequent,
                           'confidence': confidence}
            rules.append(rule)

    # recursively create associations for consequents created as combinations of current consequents
    itemset_length = len(itemeset)
    consequent_length = len(consequents[0])
    if (itemset_length > 2) and (itemset_length > consequent_length + 1):
        new_consequents = itemset_combinations(consequents)
        for rule in itemset_rules(itemeset, new_consequents, itemsets_support, minimum_confidence):
            rules.append(rule)

    return rules

def all_itemsets_rules(itemsets, itemsets_support, minimum_confidence=0.7):
    """finds relevant associations rules for each itemset in itemsets

    :param itemsets: set of itemsets from which rules will be created
    :param itemsets_support: support of all itemsets
    :param minimum_confidence: minimum confidence for an association to be relevant
    :return: all association rules created from the given itemset
    """
    rules = []
    for itemset in itemsets:
        if len(itemset) == 1: continue # no rules can be created for itemsets of size 1

        itemset_consequents = [frozenset([item]) for item in itemset]
        for rule in itemset_rules(itemset, itemset_consequents, itemsets_support, minimum_confidence):
            rules.append(rule)

    return rules
