import numpy as np
from load import Xtrain, Ytrain, Xtest, Ytest

def sigmoid(x):
    return np.where(x>=0, 1 / (1 + np.exp(-x)), np.exp(x)/(1+np.exp(x)))

W = np.random.randn(30,1) / np.sqrt(30)
b = np.array([[0]])

N = Xtrain.shape[0]
alpha = 0.05
epsilon = 10**(-15)


# Training Loop
for i in range(10000):
    Z = Xtrain@W + b
    A = np.clip(sigmoid(Z), epsilon, 1-epsilon)

    E = -1/N * (Ytrain*np.log(A) + (1-Ytrain)*np.log(1-A)).sum(axis=0)

    dZ = A-Ytrain
    dW = 2/N * Xtrain.transpose()@dZ
    db = 2/N * dZ.sum(axis=0)

    W = W - alpha * dW
    b = b - alpha * db

print(f"Training E: {E}")

# Testing
Ntest = Xtest.shape[0]

Z = Xtest@W + b
A = np.clip(sigmoid(Z), epsilon, 1-epsilon)

E = -1/Ntest * (Ytest*np.log(A) + (1-Ytest)*np.log(1-A)).sum(axis=0)

print(f"Testing E: {E}")
A = (A >= 0.5).astype(int)

accurate_classification = (A == Ytest).sum()
test_count = Ytest.shape[0]

print(f"Total Number of Test Patients: {test_count}")
print(f"Number of Correct Detections: {accurate_classification}")
print(f"Accuracy: {accurate_classification/test_count * 100}")

# Training E: [0.03348162]
# Testing E: [0.12125911]
# Total Number of Test Patients: 114
# Number of Correct Detections: 109
# Accuracy: 95.6140350877193