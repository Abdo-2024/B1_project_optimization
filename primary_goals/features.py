import numpy as np

def create_features_for_poly(X, degree):
    """
    Generates polynomial features up to a specified degree.
    Parameters:
        X (numpy.ndarray): Original feature matrix of shape (n_samples, 2).
        degree (int): Maximum polynomial degree.
    Returns:
        numpy.ndarray: Feature matrix with polynomial terms.
    """
    if degree < 1:
        raise ValueError("Degree must be at least 1.")
    features = [X]
    for d in range(2, degree + 1):
        features.append(np.power(X, d))
    return np.hstack(features)
