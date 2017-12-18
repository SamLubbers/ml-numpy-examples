"""example usage of the apriori module"""

# create fictional dataset of transactions
dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


from apriori import create_initial_itemsets
initial_itemsets = create_initial_itemsets(dataset)

from apriori import filter_itemsets
filtered_itemsets, itemsets_support = filter_itemsets(dataset, initial_itemsets, min_support=0.5)

from apriori import itemset_combinations
new_itemsets = itemset_combinations(filtered_itemsets)

from apriori import apriori
most_frequent_itemsets, support = apriori(dataset)

from apriori import itemset_rules
example_itemset = most_frequent_itemsets[8]
example_consequents = [frozenset([item]) for item in example_itemset]
example_associations = itemset_rules(example_itemset,
                                     example_consequents,
                                     support)
