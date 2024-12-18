import numpy as np

def mean_logloss(X, y_real, theta):
    """
    Calculate the Mean Log-Loss for logistic regression.

    Args:
        X (numpy.ndarray): Feature matrix, dimensions (n_samples, n_features + 1).
        y_real (numpy.ndarray): True class labels, dimensions (n_samples,).
        theta (numpy.ndarray): Logistic regression parameters, dimensions (n_features + 1,).

    Returns:
        float: Mean log-loss.
    """
    predictions = 1 / (1 + np.exp(-X @ theta))  # Predicted probabilities
    epsilon = 1e-10
    loss = -y_real * np.log(predictions + epsilon) - (1 - y_real) * np.log(1 - predictions + epsilon)
    return np.mean(loss)

def classif_error(y_real, y_pred):
    """
    Calculate classification error as a percentage.

    Args:
        y_real (numpy.ndarray): True class labels, dimensions (n_samples,).
        y_pred (numpy.ndarray): Predicted class labels, dimensions (n_samples,).

    Returns:
        float: Classification error percentage.
    """
    y_pred_class = (y_pred >= 0.5).astype(int)  # Convert probabilities to binary predictions
    error = np.mean(y_pred_class != y_real)
    return error * 100

def plot_data(X, class_labels):
    """
    Plot 2D data points with different colors for each class.

    Args:
        X (numpy.ndarray): Data points, dimensions (n_samples, 2).
        class_labels (numpy.ndarray): Class labels {1, 2}, dimensions (n_samples,).
    """
    import matplotlib.pyplot as plt

    size_markers = 20

    fig, ax = plt.subplots()
    ax.scatter(X[class_labels == 1, 0], X[class_labels == 1, 1], s=size_markers, c='red', edgecolors='black', linewidth=1.0, label='Class 1')
    ax.scatter(X[class_labels == 2, 0], X[class_labels == 2, 1], s=size_markers, c='green', edgecolors='black', linewidth=1.0, label='Class 2')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim([-2.0, 3.0])
    ax.set_ylim([-2.0, 3.0])
    ax.legend()
    ax.grid(True)

    plt.show()
