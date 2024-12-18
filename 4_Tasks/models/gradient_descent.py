import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        """
        Gradient Descent Optimizer.

        Args:
            learning_rate (float): Learning rate for parameter updates.
        """
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """
        Update model parameters using gradients.

        Args:
            params (dict): Model parameters (weights and biases).
            gradients (dict): Gradients of the loss w.r.t. parameters.
        """
        for key in params.keys():
            params[key] -= self.learning_rate * gradients[key]

        return params


def optimize_with_gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Optimize parameters using gradient descent.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
        theta (numpy.ndarray): Initial parameters.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations.

    Returns:
        numpy.ndarray: Optimized parameters.
    """
    m = len(y)

    for i in range(num_iterations):
        # Predictions
        z = np.dot(X, theta)
        predictions = 1 / (1 + np.exp(-z))

        # Compute gradient
        gradient = np.dot(X.T, (predictions - y)) / m

        # Update parameters
        theta -= learning_rate * gradient

        # Log loss for every 100 iterations
        if i % 100 == 0:
            loss = -(1 / m) * np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return theta
