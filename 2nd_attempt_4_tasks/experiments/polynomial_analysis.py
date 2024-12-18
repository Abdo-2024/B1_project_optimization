import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from models.logistic_regression import create_features_for_poly, log_regr
from models.optimizer import grad_descent
from scripts.utils import classif_error

def training_size_analysis(X_train, y_train, X_val, y_val, degree=2, learning_rate=0.1, gd_iters=1000):
    """
    Analyze model performance with varying training sample sizes.

    Args:
        X_train (numpy.ndarray): Full training feature matrix.
        y_train (numpy.ndarray): Full training labels.
        X_val (numpy.ndarray): Validation feature matrix.
        y_val (numpy.ndarray): Validation labels.
        degree (int): Polynomial degree for feature transformation.
        learning_rate (float): Learning rate for gradient descent.
        gd_iters (int): Number of gradient descent iterations.
    """
    sample_sizes = [50, 100, 200, 400]  # Subset sizes to test
    train_errors = []
    val_errors = []

    for n in sample_sizes:
        print(f"Training with {n} samples...")

        # Subset the training data
        X_train_subset = X_train[:n]
        y_train_subset = y_train[:n]

        # Generate polynomial features
        X_train_poly = create_features_for_poly(X_train_subset, degree)
        X_train_poly = np.c_[X_train_poly, np.ones(X_train_poly.shape[0])]  # Add bias term

        X_val_poly = create_features_for_poly(X_val, degree)
        X_val_poly = np.c_[X_val_poly, np.ones(X_val_poly.shape[0])]

        # Train the model
        theta_opt = grad_descent(X_train_poly, y_train_subset, learning_rate, gd_iters)

        # Calculate errors
        train_error = classif_error(y_train_subset, log_regr(X_train_poly, theta_opt))
        val_error = classif_error(y_val, log_regr(X_val_poly, theta_opt))

        train_errors.append(train_error)
        val_errors.append(val_error)

        print(f"Sample Size: {n}, Train Error: {train_error:.2f}%, Validation Error: {val_error:.2f}%")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(sample_sizes, train_errors, label="Train Error", marker='o')
    plt.plot(sample_sizes, val_errors, label="Validation Error", marker='o')
    plt.xlabel("Training Sample Size")
    plt.ylabel("Error (%)")
    plt.title("Effect of Training Sample Size on Model Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

def polynomial_degree_comparison(X_train, y_train, X_val, y_val, max_degree=5, learning_rate=0.1, gd_iters=1000):
    """
    Compare training and validation errors for various polynomial degrees.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation feature matrix.
        y_val (numpy.ndarray): Validation labels.
        max_degree (int): Maximum polynomial degree to test.
        learning_rate (float): Learning rate for gradient descent.
        gd_iters (int): Number of gradient descent iterations.

    Returns:
        int: The best polynomial degree based on validation error.
    """
    train_errors = []
    val_errors = []
    degrees = list(range(1, max_degree + 1))

    for degree in degrees:
        print(f"Testing Polynomial Degree {degree}...")

        # Generate polynomial features
        X_train_poly = create_features_for_poly(X_train, degree)
        X_train_poly = np.c_[X_train_poly, np.ones(X_train_poly.shape[0])]  # Add bias term

        X_val_poly = create_features_for_poly(X_val, degree)
        X_val_poly = np.c_[X_val_poly, np.ones(X_val_poly.shape[0])]

        # Train the model
        theta_opt = grad_descent(X_train_poly, y_train, learning_rate, gd_iters)

        # Calculate errors
        train_error = classif_error(y_train, log_regr(X_train_poly, theta_opt))
        val_error = classif_error(y_val, log_regr(X_val_poly, theta_opt))

        train_errors.append(train_error)
        val_errors.append(val_error)

        print(f"Degree: {degree}, Train Error: {train_error:.2f}%, Validation Error: {val_error:.2f}%")

    # Find the best polynomial degree (lowest validation error)
    best_degree = degrees[np.argmin(val_errors)]
    print(f"\nBest Polynomial Degree: {best_degree}")

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_errors, label="Train Error", marker='o')
    plt.plot(degrees, val_errors, label="Validation Error", marker='o')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Error (%)")
    plt.title("Training vs Validation Error for Polynomial Degrees")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_degree
