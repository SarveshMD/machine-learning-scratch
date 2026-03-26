import numpy as np

epsilon = 10**(-15)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(1/n_inputs)
        self.b = np.zeros((1, n_neurons))

    def forward(self, X):
        assert X.shape[1] == self.n_inputs, f"Input shape invalid. {X.shape[1]} != {self.n_inputs}"
        self.input = X
        Z = X@self.W + self.b
        return Z

    def backward(self, grad):
        self.dW = 1/self.input.shape[0] * self.input.transpose()@grad
        self.db = 1/self.input.shape[0] * grad.sum(axis=0).reshape(1,self.n_neurons)
        dX = grad@self.W.transpose()
        return dX

    def update(self, alpha):
        self.W -= alpha * self.dW
        self.b -= alpha * self.db

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        self.A = np.clip(np.where(X>=0, 1 / (1 + np.exp(-X)), np.exp(X)/(1+np.exp(X))), epsilon, 1-epsilon)
        return self.A

    def backward(self, grad):
        dZ = grad*(self.A*(1-self.A))
        return dZ

    def update(self, alpha):
        pass

class ReLU:
    def __init__(self):
        pass

    def forward(self, X):
        self.input = X
        self.A = np.maximum(0, X)
        return self.A

    def backward(self, grad):
        new_grad = np.copy(grad)
        new_grad[self.input <= 0] = 0
        return new_grad

    def update(self, alpha):
        pass

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)

def main():
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

    alpha = 0.05

    model = NeuralNetwork([DenseLayer(4, 5), Sigmoid(), DenseLayer(5,3), Sigmoid()])
    for i in range(10000):
        predictions = model.forward(X)
        grad = (predictions - Y)/(predictions*(1-predictions)+epsilon)
        model.backward(grad)
        model.update(alpha)

    predictions = model.forward(X)
    print("predictions: ", predictions)
    print("W2: ", model.layers[2].W)
    print("b2: ", model.layers[2].b)

if __name__ == "__main__":
    main()