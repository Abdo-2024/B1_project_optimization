import numpy as np
import matplotlib.pyplot as plt
from models.logistic_regression import LogisticRegression

def create_features_for_poly(X, degree):
    """
    Expands input features to the specified polynomial degree.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (n_samples, n_features).
    degree (int): Degree of the polynomial features.

    Returns:
    numpy.ndarray: Expanded features of shape (n_samples, n_poly_features).
    """
    if degree < 1:
        raise ValueError("Degree must be at least 1.")
    
    n_samples, n_features = X.shape
    features_poly = X

    for d in range(2, degree + 1):
        features_poly = np.concatenate([features_poly, X**d], axis=1)

    return features_poly

def mean_logloss(y_real, y_pred):
    """
    Computes the mean log loss for a binary classification problem.

    Parameters:
    y_real (numpy.ndarray): True labels, shape (n_samples,).
    y_pred (numpy.ndarray): Predicted probabilities, shape (n_samples,).

    Returns:
    float: Mean log loss.
    """
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

def classif_error(y_real, y_pred):
    """
    Computes classification error percentage.

    Parameters:
    y_real (numpy.ndarray): True labels, shape (n_samples,).
    y_pred (numpy.ndarray): Predicted labels, shape (n_samples,).

    Returns:
    float: Classification error percentage.
    """
    incorrect = np.sum(y_real != y_pred)
    return (incorrect / len(y_real)) * 100


def plot_decision_boundary(X, y, theta, degree):
    """
    Plots decision boundary for a logistic regression model with polynomial features.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (n_samples, n_features).
    y (numpy.ndarray): True labels, shape (n_samples,).
    theta (numpy.ndarray): Optimized parameters.
    degree (int): Degree of polynomial features.
    """
    # Generate a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create polynomial features for the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_poly = create_features_for_poly(grid, degree)
    grid_poly = np.concatenate((grid_poly, np.ones((grid_poly.shape[0], 1))), axis=1)  # Add bias

    # Predict probabilities
    probs = LogisticRegression(grid_poly, theta).reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.6)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label="Class 1")
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label="Class 0")
    plt.title("Decision Boundary")
    plt.legend()
    plt.show()

def shuffle_data(X, y):
    """
    Randomly shuffles the dataset.

    Parameters:
    X (numpy.ndarray): Features matrix, shape (n_samples, n_features).
    y (numpy.ndarray): Labels vector, shape (n_samples,).

    Returns:
    tuple: Shuffled X and y.
    """
    permutation = np.random.permutation(len(y))
    return X[permutation], y[permutation]

