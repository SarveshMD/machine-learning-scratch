# 5 runners, 4 factors, 3 outputs

import numpy as np
from misc import sigmoid

X = np.array([
    [2.1, 1.5, 3.0, 1.2],
    [0.5, 0.2, 1.1, 0.5],
    [2.5, 1.8, 3.5, 1.5],
    [0.8, 0.4, 0.9, 0.3],
    [3.0, 2.1, 4.1, 1.8]
])

Y = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
    [1, 0, 1]
])

N = X.shape[0]
alpha = 0.05

W1 = np.random.randn(4,4)
b1 = np.zeros((1,4))

W2 = np.random.randn(4,3)
b2 = np.zeros((1,3))

for i in range(100000):
    Z1 = X@W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1@W2 + b2
    A2 = sigmoid(Z2)

    # E = -1/N * ((Y*np.log(A2)) + (1-Y)*np.log(1-A2)).sum(axis=0) # Not necessary to calculate unless printing out
    # print(f"E{i+1}: {E}")

    dZ2 = A2 - Y
    dW2 = 1/N * A1.transpose()@dZ2
    db2 = 1/N * dZ2.sum(axis=0)

    dZ1 = (dZ2@W2.transpose()) * (A1*(1-A1))
    dW1 = 1/N * X.transpose()@dZ1
    db1 = 1/N * dZ1.sum(axis=0)

    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)

    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * db2)


E = -1/N * ((Y*np.log(A2)) + (1-Y)*np.log(1-A2)).sum(axis=0) # can calculate once, for printing out in the end
print(f"W2: {W2}\nb2: {b2}\nE: {E}\ny: {A2}\nY: {Y}")