"""example that uses logistic regression to estimate horse fatalities from colic"""

# loading dataset
import pandas as pd
dataset_train = pd.read_csv('horseColicTraining.txt', delimiter='\t', header=None)
dataset_test = pd.read_csv('horseColicTest.txt', delimiter='\t', header=None)

X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, len(dataset_train.columns)-1].values

X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, len(dataset_test.columns)-1].values
