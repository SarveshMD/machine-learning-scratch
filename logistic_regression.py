import numpy as np
from misc import sigmoid

X = np.array([
    [2.1, 1.5, 3.0, 1.2],
    [0.5, 0.2, 1.1, 0.5],
    [2.5, 1.8, 3.5, 1.5],
    [0.8, 0.4, 0.9, 0.3],
    [3.0, 2.1, 4.1, 1.8]
])

Y = np.array([1, 0, 1, 0, 1])

N = X.shape[0]
W = np.ones((4,1))
b = 0

alpha = 0.05

for i in range(10000):
    z = X@W + b

    y = sigmoid(z).flatten()
    E = -1/N * (Y*np.log(y) + (1-Y)*np.log(1-y)).sum()  # Binary Cross-Entropy Formula

    print(f"E{i+1}: {E}")

    y_Y = y-Y

    dE_dW = 2/N * X.transpose()@y_Y
    dE_db = 2/N * y_Y.sum()

    W = W - (alpha * dE_dW).reshape(4,1)
    b = b - (alpha * dE_db)

print(f"W: {W}\nb: {b}\nE: {E}\ny: {y}\nY: {Y}")