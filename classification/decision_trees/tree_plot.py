"""plotting decision tree using matplotlib annotations"""
import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_txt, point_coordinates, text_coordinates, node_type):
    create_plot.ax1.annotate(node_txt,
                 xy=point_coordinates,
                 xytext=text_coordinates,
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

def create_plot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node(node_txt='decision node',
              point_coordinates=(0.1, 0.5),
              text_coordinates=(0.5, 0.1),
              node_type=decision_node)
    plot_node(node_txt='leaf node',
          point_coordinates=(0.1, 0.8),
          text_coordinates=(0.8, 0.1),
          node_type=leaf_node)
    plt.show()