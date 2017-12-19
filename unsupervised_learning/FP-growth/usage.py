"""example usage of the fpgrowth module"""

# example tree of few nodes
from fpgrowth import Node
node = Node('x', 9, None)
node.children['y'] = Node('y', 5, None)
node.children['z'] = Node('z', 2, None)

# print simple tree
node.display_tree()

# example dataset of transactions
dataset = [['r', 'z', 'h', 'j', 'p'],
           ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
           ['z'],
           ['r', 'x', 'n', 'o', 's'],
           ['y', 'r', 'x', 'z', 'q', 't', 'p'],
           ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

# parse input
from fpgrowth import prepare_input_data
parsed_dataset = prepare_input_data(dataset)

# create headertable from dataset
from fpgrowth import create_header_table
headertable = create_header_table(parsed_dataset, 3)
