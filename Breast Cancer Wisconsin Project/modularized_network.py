from class_based_init import DenseLayer, Sigmoid, ReLU, NeuralNetwork, epsilon
from load import Xtrain, Ytrain, Xtest, Ytest
import numpy as np

# Hyperparameters:
# alpha (Learning Rate) = 0.005
# Depth = 1. 1 Hidden Layer with 16 neurons, ReLU

def main():
    N = Xtrain.shape[0]
    alpha = 0.005

    model = NeuralNetwork([DenseLayer(30,16), ReLU(),
                          DenseLayer(16,1), Sigmoid()])

    for i in range(20000):
        predictions = model.forward(Xtrain)
        grad = (predictions - Ytrain)/(predictions*(1-predictions)+epsilon)
        model.backward(grad)
        model.update(alpha)

    E = -1/N * (Ytrain*np.log(predictions) + (1-Ytrain)*np.log(1-predictions)).sum(axis=0)
    print(f"Training E: {E}\n")

    N = Xtest.shape[0]
    predictions = model.forward(Xtest)
    E = -1/N * (Ytest*np.log(predictions) + (1-Ytest)*np.log(1-predictions)).sum(axis=0)

    print(f"Testing E: {E}\n")
    predictions = (predictions >= 0.5).astype(int)

    accurate_classification = (predictions == Ytest).sum()
    test_count = Ytest.shape[0]

    print(f"Total Number of Test Patients: {test_count}")
    print(f"Number of Correct Detections: {accurate_classification}")
    print(f"Accuracy: {accurate_classification/test_count * 100}")

if __name__ == "__main__":
    main()