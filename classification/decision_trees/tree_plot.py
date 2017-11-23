"""plotting decision tree using matplotlib annotations"""

import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_txt, node_coordinates, parent_coordinates, node_type):
    create_plot.ax1.annotate(node_txt,
                             xy=parent_coordinates,
                             xytext=node_coordinates,
                             xycoords='axes fraction',
                             textcoords='axes fraction',
                             va='center',
                             ha='center',
                             bbox=node_type,
                             arrowprops=arrow_args)

def calculate_tree_leaves(my_tree):
    num_leaves = 0
    subnodes = next(iter(my_tree.values()))
    for subnode in subnodes.values():
        if type(subnode) is dict:
            num_leaves += calculate_tree_leaves(subnode)
        else:
            num_leaves += 1
    return num_leaves
    
def calcualte_tree_depth(my_tree):
    max_depth = 0
    subnodes = next(iter(my_tree.values()))
    for subnode in subnodes.values():
        if type(subnode) is dict:
            depth = 1 + calcualte_tree_depth(subnode)
        else:
            depth = 1
        if depth > max_depth:
            max_depth = depth
    return max_depth

def plot_link_label(node_coordinates, parent_coordinates, link_label):
    offset = 0.02 # apply offset to avoid printing label exactly on link
    x_mid = (parent_coordinates[0] + node_coordinates[0]) / 2.0 + offset
    y_mid = (parent_coordinates[1] + node_coordinates[1]) / 2.0
    create_plot.ax1.text(x_mid, y_mid, link_label)


def plot_tree(my_tree, parent_coordinates, link_label):
    num_leaves = calculate_tree_leaves(my_tree)
    node_label = next(iter(my_tree.keys()))
    node_coordinates = (plot_tree.x_off + (1.0 + num_leaves) / 2.0 / plot_tree.total_width, plot_tree.y_off)
    plot_link_label(node_coordinates, parent_coordinates, link_label)
    plot_node(node_label, node_coordinates, parent_coordinates, decision_node)
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_depth

    subnodes = next(iter(my_tree.values()))
    for subnode_name in subnodes.keys():
        if type(subnodes[subnode_name]) is dict:
            plot_tree(subnodes[subnode_name], node_coordinates, str(subnode_name))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_width
            plot_node(subnodes[subnode_name], (plot_tree.x_off, plot_tree.y_off), node_coordinates, leaf_node)
            plot_link_label((plot_tree.x_off, plot_tree.y_off), node_coordinates, str(subnode_name))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_depth
    
    
def create_plot(my_tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False, xticks=[], yticks=[]) #no ticks
    plot_tree.total_width = calculate_tree_leaves(my_tree)
    plot_tree.total_depth = calcualte_tree_depth(my_tree)
    plot_dimensions = (1, 1)
    root_coordinates = (plot_dimensions[0]/2,plot_dimensions[1]) # initialize root node in the middle center and top of the graph
    plot_tree.x_off = -root_coordinates[0]/plot_tree.total_width # offset of the x coordinate
    plot_tree.y_off = root_coordinates[1] # offset of the y coordinate
    plot_tree(my_tree, root_coordinates, '')
    plt.show()
    