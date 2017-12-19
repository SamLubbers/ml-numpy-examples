"""example usage of the fpgrowth module"""

# example tree of few nodes
from fpgrowth import Node
node = Node('x', 9, None)
node.children['y'] = Node('y', 5, None)
node.children['z'] = Node('z', 2, None)

# print simple tree
node.display_tree()