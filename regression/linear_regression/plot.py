import numpy as np
import matplotlib.pyplot as plt
from usage import X, y, X_sorted, y_hat, y_hat_lwlr

fig = plt.figure()

ax = fig.add_subplot(111)
# scatter data points
ax.scatter(X[:, 1], y, color='gray')

# plot linear regression line
ax.plot(X_sorted[:, 1], y_hat, color='blue')

# plot local weighted linear regression line
ax.plot(X_sorted[:, 1], y_hat_lwlr, color='red')

plt.title('simple linear regression')
plt.xlabel('feature')
plt.ylabel('target value')

# plotting multiple coefficients obtained through ridge regression
from usage import abalone_multiple_ridge_weights

fig2 = plt.figure()
ax2 = fig2.add_subplot(311)
ax2.plot(abalone_multiple_ridge_weights)

plt.title('ridge regression')
plt.xlabel('iterations')
plt.ylabel('weights')
# plotting multiple coefficients obtained through stagewise regression
from usage import abalone_multiple_stagewise_weights

ax3 = fig2.add_subplot(313)
ax3.plot(abalone_multiple_stagewise_weights)
plt.title('stagewise forward regression')
plt.xlabel('iterations')
plt.ylabel('weights')

plt.show()