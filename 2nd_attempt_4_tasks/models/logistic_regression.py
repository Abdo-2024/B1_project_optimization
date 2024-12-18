import numpy as np

def log_regr(X, theta):
    """
    Logistic regression prediction function.

    Args:
        X (numpy.ndarray): Feature matrix, dimensions (n_samples, n_features + 1), includes bias term.
        theta (numpy.ndarray): Logistic regression parameters, dimensions (n_features + 1,).

    Returns:
        numpy.ndarray: Predicted probabilities for class 1, dimensions (n_samples,).
    """
    return 1 / (1 + np.exp(-X @ theta))  # Sigmoid activation

def create_features_for_poly(X, degree):
    """
    Generate polynomial features up to a given degree.

    Args:
        X (numpy.ndarray): Original feature matrix, dimensions (n_samples, 2).
        degree (int): Degree of the polynomial.

    Returns:
        numpy.ndarray: Feature matrix with polynomial features, dimensions (n_samples, num_polynomial_features).
    """
    from itertools import combinations_with_replacement

    n_samples, n_features = X.shape
    poly_features = []

    # Generate polynomial features for all degrees
    for d in range(1, degree + 1):
        for terms in combinations_with_replacement(range(n_features), d):
            feature = np.prod([X[:, t] for t in terms], axis=0)
            poly_features.append(feature)

    return np.vstack(poly_features).T
