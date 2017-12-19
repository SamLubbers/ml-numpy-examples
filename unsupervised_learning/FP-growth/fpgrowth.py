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
