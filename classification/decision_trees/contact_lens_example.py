"""using decision tree to aid in the prescription of contact lense type"""
import pandas as pd
from decision_trees import create_tree
from tree_plot import create_plot

# load the dataset
dataset = pd.read_csv('lenses.txt', delimiter='\t', header=None)

dataset.columns = ['age',
                   'prescript',
                   'astigmatic',
                   'tear_rate',
                   'needs_lenses']

lenses_tree = create_tree(dataset)

create_plot(lenses_tree)