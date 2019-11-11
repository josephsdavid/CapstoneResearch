import matplotlib.pyplot as plt
import numpy as np



x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

plt.plot(x, label='x')
plt.plot(y, label='y')
plt.title('Our two temporal sequences')
plt.legend()
plt.show()

from dtw import dtw
l2_norm = lambda x, y: (x-y)**2

dist, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=l2_norm)
dist
