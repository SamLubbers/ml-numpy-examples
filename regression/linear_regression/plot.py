import numpy as np
import matplotlib.pyplot as plt
from usage import X, y, X_sorted, y_hat, y_hat_lwlr

# scatter data points
plt.scatter(X[:, 1], y, color='gray')

# plot linear regression line
plt.plot(X_sorted[:, 1], y_hat, color='blue')

# plot local weighted linear regression line
plt.plot(X_sorted[:, 1], y_hat_lwlr, color='red')
plt.show()