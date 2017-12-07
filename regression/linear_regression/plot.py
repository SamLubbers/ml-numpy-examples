import numpy as np
import matplotlib.pyplot as plt
from usage import X, y, X_sorted, y_hat

plt.scatter(X[:, 1], y)
plt.plot(X_sorted[:, 1], y_hat)
plt.show()