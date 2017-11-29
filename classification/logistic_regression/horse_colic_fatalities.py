"""example that uses logistic regression to estimate horse fatalities from colic"""

# loading dataset
import pandas as pd
dataset_train = pd.read_csv('horseColicTraining.txt', delimiter='\t', header=None)
dataset_test = pd.read_csv('horseColicTest.txt', delimiter='\t', header=None)

X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, len(dataset_train.columns)-1].values

X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, len(dataset_test.columns)-1].values

# obtaining optimal weight values
from logistic_regression import optimal_weights_stochastic_ascent
weights = optimal_weights_stochastic_ascent(X_train, y_train)

# making predictions
from logistic_regression import classify_logistic_regression

wrong_guesses = 0
for test_vector, test_label in zip(X_test, y_test):
    predicted_label = classify_logistic_regression(test_vector, weights)
    if predicted_label != test_label:
        wrong_guesses += 1

error_rate = wrong_guesses/len(X_test)

print('\nlogistic regression classifier has an error rate of: %f' % error_rate)