import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-1*z))

# print(sigmoid(-50))
# print(sigmoid(0))
# print(sigmoid(99))
