import numpy as np
import matplotlib.pyplot as plt

def create_data(n_samples):
    """
    Generates synthetic data for two classes using predefined Gaussians.
    Parameters:
        n_samples (int): Total number of samples.
    Returns:
        X (numpy.ndarray): Feature matrix of shape (n_samples, 2).
        class_labels (numpy.ndarray): Labels (1 or 2) for the samples.
    """
    n_classes = 2
    n_samples_per_class = n_samples // n_classes

    # Gaussian parameters for Class 1
    x_mu1 = np.array([[1.3, -0.6], [-0.3, 1.2], [-0.4, -0.6]])
    x_var1 = [np.array([[0.3, 0.0], [0.0, 0.15]]),
              np.array([[0.15, 0.05], [0.05, 0.25]]),
              np.array([[0.15, -0.05], [-0.05, 0.15]])]

    # Gaussian parameters for Class 2
    x_mu2 = np.array([[1.4, 0.9], [0.9, 2.0]])
    x_var2 = [np.array([[0.4, 0.1], [0.1, 0.15]]),
              np.array([[0.2, -0.1], [-0.1, 0.1]])]

    # Data generation logic
    X, class_labels = [], []
    for c, (x_mu, x_var) in enumerate([(x_mu1, x_var1), (x_mu2, x_var2)], start=1):
        for cluster in range(len(x_mu)):
            samples = np.random.multivariate_normal(x_mu[cluster], x_var[cluster], n_samples_per_class // len(x_mu))
            X.append(samples)
            class_labels.append(np.full(samples.shape[0], c))

    X = np.vstack(X)
    class_labels = np.hstack(class_labels)

    # Shuffle samples
    shuffle_idx = np.random.permutation(X.shape[0])
    return X[shuffle_idx], class_labels[shuffle_idx]

def plot_data(X, class_labels):
    """
    Visualizes the data with separate colors for each class.
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, 2).
        class_labels (numpy.ndarray): Labels (1 or 2) for the samples.
    """
    plt.scatter(X[class_labels == 1][:, 0], X[class_labels == 1][:, 1], c='red', label='Class 1')
    plt.scatter(X[class_labels == 2][:, 0], X[class_labels == 2][:, 1], c='green', label='Class 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()
