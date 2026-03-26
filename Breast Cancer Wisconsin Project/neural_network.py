import numpy as np
from load import Xtrain, Ytrain, Xtest, Ytest

def sigmoid(x):
    return np.where(x>=0, 1 / (1 + np.exp(-x)), np.exp(x)/(1+np.exp(x)))

HIDDEN_WIDTH = 8 # Width of Hidden Layer, i.e, Number of Neurons in Hidden Layer

W1 = np.random.randn(30,HIDDEN_WIDTH) / np.sqrt(30*HIDDEN_WIDTH)
b1 = np.zeros((1,HIDDEN_WIDTH))

W2 = np.random.randn(HIDDEN_WIDTH,1) / np.sqrt(HIDDEN_WIDTH)
b2 = np.array([[0]])

N = Xtrain.shape[0]
alpha = 0.05
epsilon = 10**(-15)


# Training Loops
for i in range(10000):
    Z1 = Xtrain@W1 + b1
    A1 = np.clip(sigmoid(Z1), epsilon, 1-epsilon)

    Z2 = A1@W2 + b2
    A2 = np.clip(sigmoid(Z2), epsilon, 1-epsilon)

    dZ2 = A2-Ytrain
    dW2 = 1/N * A1.transpose()@dZ2
    db2 = 1/N * dZ2.sum(axis=0).reshape(1,1)

    dZ1 = (dZ2@W2.transpose()) * (A1 * (1-A1))
    dW1 = 1/N * Xtrain.transpose()@dZ1
    db1 = (1/N * dZ1.sum(axis=0)).reshape(1,HIDDEN_WIDTH)

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

E = -1/N * (Ytrain*np.log(A2) + (1-Ytrain)*np.log(1-A2)).sum(axis=0)
print(f"\nTraining E: {E}")

# Testing
Ntest = Xtest.shape[0]

Z1 = Xtest@W1 + b1
A1 = np.clip(sigmoid(Z1), epsilon, 1-epsilon)

Z2 = A1@W2 + b2
A2 = np.clip(sigmoid(Z2), epsilon, 1-epsilon)

E = -1/Ntest * (Ytest*np.log(A2) + (1-Ytest)*np.log(1-A2)).sum(axis=0)

print(f"Testing E: {E}\n")
A2 = (A2 >= 0.5).astype(int)

accurate_classification = (A2 == Ytest).sum()
test_count = Ytest.shape[0]

print(f"Number of Neurons in Hidden Layer: {HIDDEN_WIDTH}")
print(f"Total Number of Test Patients: {test_count}")
print(f"Number of Correct Detections: {accurate_classification}")
print(f"Accuracy: {accurate_classification/test_count * 100}")

# Training E: [0.8162]
# Testing E: [0.12125911]
# Total Number of Test Patients: 114
# Number of Correct Detections: 109
# Accuracy: 95.6140350877193