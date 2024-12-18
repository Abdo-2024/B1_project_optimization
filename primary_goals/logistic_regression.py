import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    Parameters:
        z (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Sigmoid-transformed values.
    """
    return 1 / (1 + np.exp(-z))

def log_regr(X, theta):
    """
    Logistic regression model: computes probabilities.
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        theta (numpy.ndarray): Model weights of shape (n_features, 1).
    Returns:
        numpy.ndarray: Predicted probabilities of shape (n_samples, 1).
    """
    return sigmoid(np.dot(X, theta))

def mean_logloss(X, y, theta):
    """
    Computes the mean log-loss.
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): True labels (0 or 1) of shape (n_samples, 1).
        theta (numpy.ndarray): Model weights of shape (n_features, 1).
    Returns:
        float: Mean log-loss.
    """
    y_pred = log_regr(X, theta)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def grad_descent(X, y, learning_rate, iters):
    """
    Performs gradient descent for logistic regression.
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): True labels of shape (n_samples, 1).
        learning_rate (float): Step size.
        iters (int): Number of iterations.
    Returns:
        numpy.ndarray: Optimized weights (theta).
    """
    theta = np.zeros((X.shape[1], 1))
    m = X.shape[0]
    loss_history = []

    for _ in range(iters):
        y_pred = log_regr(X, theta)
        gradient = np.dot(X.T, (y_pred - y)) / m
        theta -= learning_rate * gradient
        loss_history.append(mean_logloss(X, y, theta))  # Track loss

    return theta, loss_history

def classif_error(y_real, y_pred):
    """
    Calculates classification error.
    Parameters:
        y_real (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
    Returns:
        float: Error percentage.
    """
    return np.mean(y_real != y_pred) * 100
