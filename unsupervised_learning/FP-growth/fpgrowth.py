"""FP-growth algorithm. Uses an FP-tree data structure to quickly find most frequent itemsets"""

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
    """creates the header table, necessary for the creation of the fp-tree"""
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
