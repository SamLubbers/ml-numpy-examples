"""FP-growth algorithm. Uses an FP-tree data structure to quickly find most frequent itemsets"""
import operator

class Node(object):
    """node of an fp-tree"""
    def __init__(self, item, frequency, parent_node):
        self.item = item
        self.frequency = frequency
        self.parent = parent_node
        self.node_link = None
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
    :param min_support: minimum number of times each item must appear to be kept in the header table
    """
    header_table = {}
    for transaction in dataset:
        for item in transaction:
            header_table[item] = header_table.setdefault(item, 0) + dataset[transaction]

    for item in list(header_table.keys()):
        if header_table[item] < min_support:
            del header_table[item]

    for item in list(header_table.keys()):
        header_table[item] = {'count': header_table[item], 'node_link': None}

    return header_table

def create_tree(dataset, header_table):
    """creates the FP-tree given a dataset of transaction and a header_table for that dataset

    :param dataset: dictionary with frozensets of instances as keys and their frequency as values
    :param header_table: dictionary containing the frequency of each item, without node_link to first node of that item
    :return: FP-tree and header_table, with item count and node_link to first node of that item
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

    return tree, header_table


def update_tree(ordered_transaction, parent_node, header_table, count):
    """recursively updates the FP-tree with each item in ordered_transaction

    :param ordered_transaction: list of items in a transaction, or transaction subset, sorted by frequency of occurance
    :param parent_node: parent node of first item of the ordered_transaction
                        if ordered_transaction is a full transaction the parent node will be the root node
                        if ordered_transaction is a subset, parent node is the previous node in the full transaction
    :param header_table: dictionary containing the frequency of each item and node_link to first node of that item
    :param count: number of times this transaction occurs in our dataset
    """
    item = ordered_transaction[0] # create Node for first item in the ordered_transaction (most frequent item)
    if item in parent_node.children:
        parent_node.children[item].frequency += 1
    else:
        new_node = Node(item=item,
                        frequency=count,
                        parent_node=parent_node)
        parent_node.children[item] = new_node

        update_node_link(header_table, new_node)
    # recursively created nodes for remaining items in the transaction
    if len(ordered_transaction) > 1:
        update_tree(ordered_transaction[1:],parent_node.children[item], header_table, count)

def update_node_link(header_table, item_node):
    """updates the node_link of the header_table or a Node with a reference to the new Node of a certain item

    :param header_table: dictionary containing the frequency of each item and node_link to first node of that item
    :param item_node: node of the item that has been placed in the fp-tree
    """
    item = item_node.item
    # if it is the first Node created of that item populate node link in header_table
    if header_table[item]['node_link'] is None:
        header_table[item]['node_link'] = item_node

    # else, populate last node of this item that
    else:
        node = header_table[item]['node_link']
        while node.node_link is not None:
            node = node.node_link

        node.node_link = item_node

def fp_tree(dataset, min_support):
    """creates a complete fp_tree given the parsed dataset and minimum support

    :param dataset: dictionary with frozensets of instances as keys and their frequency as values
    :param min_support: minimum number of times each item must appear to be kept in the header table
    :return: FP-tree and header table, with item count and node_link to first node of that item
    """
    header_table = create_header_table(dataset, min_support)

    if not header_table: return None, None # header table must not be empty to proceed

    tree, header_table = create_tree(dataset, header_table)
    return tree, header_table

def ascend_tree(node, prefix_path):
    """recursively scan parent nodes of node and add them to prefix path"""
    if node.parent is not None:
        prefix_path.append(node.item)
        ascend_tree(node.parent, prefix_path)

def conditional_pattern_base(item, header_table):
    """find the conditional pattern base for all nodes of item in the fp-tree

    :param item: item of which we want to obtain the conditional pattern base
    :param header_table: header_table associated with the fp-tree form which we are obtaining the prefix paths
    :return: dictionary with conditional pattern bases of the item in node_link
    """
    cpb = {}
    node = header_table[item]['node_link']
    while node is not None:
        prefix_path = []
        ascend_tree(node, prefix_path) # construct prefix path for this node
        prefix_path = prefix_path[1:] # eliminate first item as it corresponds to current node

        # add prefix path to conditional pattern base for this item
        if prefix_path:
            cpb[frozenset(prefix_path)] = node.frequency

        # find prefix path for next node of the same item
        node = node.node_link

    return cpb

def mine_tree(fptree, header_table, min_support, prefix=set([])):
    """mines an FP-tree to find frequent itemsets

    :param prefix: base items used to create a new frequent itemset.
                   Every time mine_tree is recursively called on a conditional fp-tree for variable 'x', this variable
                   is added as the prefix of the new itemset on the next recursion

    """
    frequent_itemsets = []
    # list of items in header table sorted from less frequent to most frequent
    ordered_items = [item_count[0] for item_count in sorted(header_table.items(), key=lambda p: p[1]['count'])]

    for item in ordered_items:
        new_itemset = prefix.copy()
        new_itemset.add(item)
        frequent_itemsets.append(new_itemset)
        cpb = conditional_pattern_base(item, header_table)
        conditional_tree, header_table_item = fp_tree(cpb, min_support)

        # recursively mine each new conditional_tree, stop when you run out of items in conditional tree
        if header_table_item:
            new_frequent_itemests = mine_tree(conditional_tree, header_table_item, min_support, prefix=new_itemset)
            for itemset in new_frequent_itemests:
                frequent_itemsets.append(itemset)

    return frequent_itemsets

def find_frequent_itemsets(dataset, min_support):
    """find frequent itemsets from a dataset of transactions by creating an fp-tree and mining that tree

    this function combines all other functions in the fpgrowth module
    :param dataset: list of transaction of items
    :param min_support: minimum number of times each item must appear to be relevant to a frequent itemset
    :return: frequent itemsets from the dataset
    """
    dataset = prepare_input_data(dataset)
    fptree, header_table = fp_tree(dataset, min_support)
    frequent_itemsets = mine_tree(fptree, header_table, min_support)
    return frequent_itemsets
