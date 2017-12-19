"""FP-growth algorithm. Uses an FP-tree data structure to quickly find most frequent itemsets"""
import operator

class Node(object):
    """node of an fp-tree"""
    def __init__(self, item, frequency, parent_node):
        self.item = item
        self.frequency = frequency
        self.parent = parent_node
        self.children = {}

    def display_tree(self, indent=1):
        """displays subtree containing this FPNode and all subnodes"""
        print(' '*indent+self.item+' '+str(self.frequency))
        for child in self.children.values():
            child.display_tree(indent=indent+1)

def prepare_input_data(dataset):
    """converts a list of transactions to the correct format to build an fp-tree"""
    data = {}
    for transaction in dataset:
        data[frozenset(transaction)] = data.setdefault(frozenset(transaction), 0) + 1

    return data

def create_header_table(dataset, min_support):
    """creates the header table, necessary for the creation of the fp-tree

    :param dataset: dictionary with frozensets of instances as keys and their frequency as values
    """
    header_table = {}
    for transaction in dataset:
        for item in transaction:
            header_table[item] = header_table.setdefault(item, 0) + dataset[transaction]

    for item in list(header_table.keys()):
        if header_table[item] < min_support:
            del header_table[item]

    for item in list(header_table.keys()):
        header_table[item] = {'count': header_table[item], 'pointer': None}

    return header_table

def create_tree(dataset, header_table):
    """creates the FP-tree given a dataset of transaction and a header_table for that dataset

    :param dataset: dictionary with frozensets of instances as keys and their frequency as values
    :param header_table: dictionary containing the frequency of each item, without pointers to nodes
    :return: FP-tree and header_table, with pointers to nodes
    """
    # tree main node
    tree = Node('Null set', 1, None)

    # sort items of transactions by relevance and populate node
    frequent_items = set(header_table.keys())
    for transaction, count in dataset.items():
        # each transaction is converted to a dict that holds the frequency of each item in that transaction
        transaction_items_count = {}
        for item in transaction:
            if item in frequent_items:
                transaction_items_count[item] = header_table[item]['count']

        if transaction_items_count: # only proceed if transaction_items_count is not empty
            # sort items in transaction by relevance
            ordered_transaction = [item_count[0] for item_count in sorted(transaction_items_count.items(),
                                                                          key=operator.itemgetter(1),
                                                                          reverse=True)]

            update_tree(ordered_transaction, tree, header_table, count)

    return tree


def update_tree(ordered_transaction, parent_node, header_table, count):
    """recursively updates the FP-tree with each item in ordered_transaction

    :param ordered_transaction: list of items in a transaction, or transaction subset, sorted by frequency of occurance
    :param parent_node: parent node of first item of the ordered_transaction
                        if ordered_transaction is a full transaction the parent node will be the root node
                        if ordered_transaction is a subset, parent node is the previous node in the full transaction
    :param header_table: dictionary containing the frequency of each item and pointers to nodes
    :param count: number of times this transaction occurs in our dataset
    """
    item = ordered_transaction[0] # create Node for first item in the ordered_transaction (most frequent item)
    if item in parent_node.children:
        parent_node.children[item].frequency += 1
    else:
        parent_node.children[item] = Node(item=item,
                                          frequency=count,
                                          parent_node=parent_node)

    # recursively created nodes for remaining items in the transaction
    if len(ordered_transaction) > 1:
        update_tree(ordered_transaction[1:],parent_node.children[item], header_table, count)
