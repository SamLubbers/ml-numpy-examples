import numpy as np
import matplotlib.pyplot as plt
from horse_colic_fatalities import y_test, y_test_predictions_aggregate

y_test = y_test.reshape(len(y_test))
false_positive_step = 1 / np.sum(np.array((y_test == -1)))
true_positive_step = 1 / np.sum(np.array((y_test == 1)))

# sort instances from highest prediction threshold to lowest
prediction_indices = np.array(-(y_test_predictions_aggregate.T)).argsort()

current = (0,0) # start point of our ROC curve when threshold is maximum

# loop over all prediction threshold, from highest to lowest
# as threshold decreases, each instance is classified as positive
# if correctly classified as positive the true positive rate increases
# if classified as false positive the false positive rate increases
for index in prediction_indices.tolist()[0]:
    if y_test[index] == 1:
        true_positive_rate_increase = true_positive_step
        false_positive_rate_increase = 0.0
        
    elif y_test[index] == -1:
        false_positive_rate_increase = false_positive_step
        true_positive_rate_increase = 0.0
        
    plt.plot([current[0], current[0] + false_positive_rate_increase],
             [current[1], current[1] + true_positive_rate_increase],
             c='b')
    
    current = (current[0] + false_positive_rate_increase, current[1] + true_positive_rate_increase)
    
plt.plot([0, 1], [0, 1], 'b--') # roc curve for random classification

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.axis([0, 1, 0, 1])
plt.show()

