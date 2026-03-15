import numpy as np

x1 = np.array([2, 3, 5, 8]) # Hours Trained / week
x2 = np.array([1, 2, 4, 5]) # Rest Days / week
Y = np.array([10, 15, 25, 38]) # Performance score / week

W = np.array([0, 0])
b = 0
alpha = 0.01

X = np.column_stack([x1,x2])
N = X.shape[0]

for i in range(10000):
    y = X@W + b

    y_Y = (y-Y)
    E = 1/N * y_Y**2

    dE_dW = 2/N * X.transpose()@y_Y
    dE_db = 2/N * y_Y.sum()

    W = W - (alpha * dE_dW)
    b = b - (alpha * dE_db)

print(f"W: {W}\nb: {b}\nE: {E}\ny: {y}\nY: {Y}")