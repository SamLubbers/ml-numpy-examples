import numpy as np
import matplotlib.pyplot as plt
from usage import X, y, X_sorted, y_hat, y_hat_lwlr, abalone_multiple_weights

fig = plt.figure()

ax = fig.add_subplot(111)
# scatter data points
ax.scatter(X[:, 1], y, color='gray')

# plot linear regression line
ax.plot(X_sorted[:, 1], y_hat, color='blue')

# plot local weighted linear regression line
ax.plot(X_sorted[:, 1], y_hat_lwlr, color='red')

# plotting multiple coefficients obtained through ridge regression
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(abalone_multiple_weights)

plt.show()