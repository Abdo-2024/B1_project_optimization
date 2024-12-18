import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, max_iter=1000):
        """
        Multi-layer Perceptron for binary classification.

        Args:
            layers (list): List containing the number of neurons in each layer.
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations for training.
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.params = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases for all layers."""
        np.random.seed(42)
        for i in range(1, len(self.layers)):
            self.params[f'W{i}'] = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01
            self.params[f'b{i}'] = np.zeros((self.layers[i], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        A = self.sigmoid(Z)
        return A * (1 - A)

    def forward_propagation(self, X):
        """Perform forward propagation through the network."""
        A = X
        cache = {'A0': X}
        for i in range(1, len(self.layers)):
            Z = np.dot(self.params[f'W{i}'], A) + self.params[f'b{i}']
            A = self.sigmoid(Z)
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = A
        return A, cache

    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss."""
        m = y_true.shape[1]
        loss = -(1 / m) * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss

    def backward_propagation(self, y_true, cache):
        """Perform backward propagation and compute gradients."""
        gradients = {}
        m = y_true.shape[1]
        A_final = cache[f'A{len(self.layers) - 1}']
        dA = -(np.divide(y_true, A_final + 1e-15) - np.divide(1 - y_true, 1 - A_final + 1e-15))

        for i in reversed(range(1, len(self.layers))):
            dZ = dA * self.sigmoid_derivative(cache[f'Z{i}'])
            gradients[f'dW{i}'] = (1 / m) * np.dot(dZ, cache[f'A{i - 1}'].T)
            gradients[f'db{i}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.params[f'W{i}'].T, dZ)

        return gradients

    def update_parameters(self, gradients):
        """Update weights and biases using gradient descent."""
        for i in range(1, len(self.layers)):
            self.params[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.params[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']

    def fit(self, X, y):
        """Train the neural network using gradient descent."""
        for i in range(self.max_iter):
            y_pred, cache = self.forward_propagation(X)
            loss = self.compute_loss(y, y_pred)
            gradients = self.backward_propagation(y, cache)
            self.update_parameters(gradients)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, X):
        """Predict binary labels for input data."""
        y_pred, _ = self.forward_propagation(X)
        return (y_pred >= 0.5).astype(int)
