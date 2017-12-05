"""example that uses logistic regression to estimate horse fatalities from colic"""

# loading dataset
import pandas as pd
dataset_train = pd.read_csv('horseColicTraining2.txt', delimiter='\t', header=None)
dataset_test = pd.read_csv('horseColicTest2.txt', delimiter='\t', header=None)

X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1:].values

X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, -1:].values

# making predictions on the test set
from adaboost import adaboost_classify
iterations = 50
test_set_predictions = adaboost_classify(X_train, y_train, X_test, iterations)

# number of incorrect predictions
incorrect_predictions = sum(test_set_predictions != y_test)
num_test_instances = len(y_test)
print('adaboost made %d/%d incorrect predictions' % 
     (incorrect_predictions.item(0), num_test_instances))

# error rate
error_rate = incorrect_predictions/num_test_instances
print('the error rate with %d iterations is %f' % (iterations, error_rate))