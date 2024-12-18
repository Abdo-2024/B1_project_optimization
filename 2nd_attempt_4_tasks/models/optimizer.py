"""
import numpy as np

def grad_descent(X_train, y_train, learning_rate, iters_total):
    
    Implements Gradient Descent for optimizing logistic regression parameters.

    Args:
        X_train (numpy.ndarray): Training data matrix of dimensions (n_samples, n_features + 1), includes bias term.
        y_train (numpy.ndarray): Vector of true class labels, dimensions (n_samples,).
        learning_rate (float): Learning rate for gradient descent.
        iters_total (int): Total number of gradient descent iterations.

    Returns:
        numpy.ndarray: Optimized parameters /( /theta /), dimensions (n_features + 1,).
    
    # Initialize parameters (theta) to zero
    theta = np.zeros(X_train.shape[1])

    # Perform gradient descent
    for _ in range(iters_total):
        # Compute predictions using the current parameters
        predictions = 1 / (1 + np.exp(-X_train @ theta))  # Sigmoid function

        # Compute gradient of the loss function w.r.t. theta
        gradient = (1 / len(y_train)) * (X_train.T @ (predictions - y_train))

        # Update parameters in the direction of the negative gradient
        theta -= learning_rate * gradient

    return theta
"""

# New implementation of grad_descent function
import numpy as np

def grad_descent(X, y, learning_rate=0.1, iters_total=1000, random_init=True):
    """
    Perform gradient descent to optimize logistic regression parameters.

    Args:
        X (numpy.ndarray): Feature matrix, dimensions (n_samples, n_features).
        y (numpy.ndarray): Labels {0, 1}, dimensions (n_samples,).
        learning_rate (float): Learning rate for gradient descent.
        iters_total (int): Total number of iterations.
        random_init (bool): If True, initialize theta randomly. Otherwise, initialize to zeros.

    Returns:
        numpy.ndarray: Optimized parameters theta.
    """
    n_samples, n_features = X.shape

    # Initialize theta
    if random_init:
        theta = np.random.randn(n_features) * 0.01  # Small random values
    else:
        theta = np.zeros(n_features)

    # Gradient Descent Loop
    for i in range(iters_total):
        predictions = 1 / (1 + np.exp(-X @ theta))  # Sigmoid function
        gradient = X.T @ (predictions - y) / n_samples
        theta -= learning_rate * gradient

    return theta

