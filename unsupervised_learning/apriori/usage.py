"""example usage of the apriori module"""

# create fictional dataset of transactions
dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

from apriori import create_initial_itemsets
initial_itemsets = create_initial_itemsets(dataset)