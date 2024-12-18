import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Imports for extension 
from mpl_toolkits.mplot3d import Axes3D
from models.logistic_regression import log_regr, create_features_for_poly
import numpy as np
import matplotlib.pyplot as plt

from models.logistic_regression import log_regr, create_features_for_poly
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, theta, class_labels, degree, marker_size=10):
    """
    Plots the decision boundary learned by the model.

    Args:
        X (numpy.ndarray): Original data points (without bias or polynomial features), dimensions (n_samples, 2).
        theta (numpy.ndarray): Optimized parameters, dimensions (n_features + 1,).
        class_labels (numpy.ndarray): Class labels {1, 2}, dimensions (n_samples,).
        degree (int): Degree of polynomial features used.
        marker_size (int): Size of the data points for better visibility of the contour.
    """
    # Generate a grid of points spanning the range of X
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 400), np.linspace(x2_min, x2_max, 200))

    # Flatten the grid and create polynomial features
    grid_points = np.c_[x1.ravel(), x2.ravel()]
    grid_points_poly = create_features_for_poly(grid_points, degree)
    grid_points_poly = np.concatenate((grid_points_poly, np.ones((grid_points_poly.shape[0], 1))), axis=1)  # Add bias

    # Predict probabilities for each point on the grid
    probabilities = log_regr(grid_points_poly, theta).reshape(x1.shape)

    # Plot the decision boundary
    plt.contourf(x1, x2, probabilities, levels=[0, 0.5, 1], colors=["green", "red"], alpha=0.3)

    # Remap class labels {1, 2} to {0, 1} for plotting
    class_labels_plot = (class_labels == 2).astype(int)

    # Overlay the scatter plot for data points
    plt.scatter(X[class_labels_plot == 0, 0], X[class_labels_plot == 0, 1], c='red', edgecolors='black', label='Class 1', s=marker_size)
    plt.scatter(X[class_labels_plot == 1, 0], X[class_labels_plot == 1, 1], c='green', edgecolors='black', label='Class 2', s=marker_size)

    # Add titles and labels
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    plt.ylim([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])
    plt.title(f'Decision Boundary (Degree {degree})')
    plt.legend()
    plt.grid(True)


################################################# Extension #################################################
def plot_3d_sigmoid_surface(X, class_labels, theta, degree=1):
    """
    Plot the 3D sigmoid surface representing predicted probabilities.

    Args:
        X (numpy.ndarray): Original data points, dimensions (n_samples, 2).
        class_labels (numpy.ndarray): True class labels {1, 2}.
        theta (numpy.ndarray): Optimized logistic regression parameters.
        degree (int): Degree of polynomial features used for training.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from models.logistic_regression import create_features_for_poly, log_regr
    
    # Generate a grid of points spanning the range of X
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

    # Flatten the grid and create polynomial features
    grid_points = np.c_[x1.ravel(), x2.ravel()]
    grid_points_poly = create_features_for_poly(grid_points, degree)
    grid_points_poly = np.c_[grid_points_poly, np.ones(grid_points_poly.shape[0])]  # Add bias term

    # Compute probabilities using the logistic regression model
    probabilities = log_regr(grid_points_poly, theta).reshape(x1.shape)

    # Compute predicted probabilities for actual data points
    X_poly = create_features_for_poly(X, degree)
    X_poly = np.c_[X_poly, np.ones(X_poly.shape[0])]  # Add bias term
    predicted_probs = log_regr(X_poly, theta)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sigmoid surface
    ax.plot_surface(x1, x2, probabilities, cmap="viridis", alpha=0.8)

    # Plot Class 1 points (green) and Class 2 points (red) projected on the sigmoid surface
    class_1 = X[class_labels == 1]
    class_2 = X[class_labels == 2]
    probs_class_1 = predicted_probs[class_labels == 1]
    probs_class_2 = predicted_probs[class_labels == 2]

    ax.scatter(class_1[:, 0], class_1[:, 1], probs_class_1, c='green', label='Class 1', s=10, depthshade=False)
    ax.scatter(class_2[:, 0], class_2[:, 1], probs_class_2, c='red', label='Class 2', s=10, depthshade=False)

    # Labels and Title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Predicted Probability for Class 2')
    ax.set_title("3D Visualization of Sigmoid Surface with Projected Data Points")

    plt.legend()
    plt.tight_layout()
    plt.show()



