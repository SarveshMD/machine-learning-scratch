# 5 runners, 4 factors, 3 probabilities

# factors: strength, speed, stamina, hunger (craze, fire, guts, determination) (made up)
# probabilities: qualification, injury risk, podium finish

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

N = X.shape[0] # number of runners
W = np.ones((4,3))
b = np.zeros((1,3))

alpha = 0.05

for i in range(100000):
    z = X@W + b
    y = sigmoid(z)

    y_Y = y-Y

    E = -1/N * (Y * np.log(y) + (1-Y) * np.log(1-y)).sum(axis=0)
    # print(f"E{i+1}: {E}")

    dE_dW = 2/N * X.transpose()@y_Y
    dE_db = 2/N * y_Y.sum(axis=0)

    W = W - (alpha * dE_dW)
    b = b - (alpha * dE_db)

print(f"W: {W}\nb: {b}\nE: {E}\ny: {y}\nY: {Y}")