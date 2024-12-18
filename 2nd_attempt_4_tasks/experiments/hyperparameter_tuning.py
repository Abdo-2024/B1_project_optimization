import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

""""
from models.logistic_regression import log_regr
from models.optimizer import grad_descent
from scripts.utils import classif_error
import numpy as np
import matplotlib.pyplot as plt

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    
    Perform hyperparameter tuning by testing various learning rates and iteration counts.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation feature matrix.
        y_val (numpy.ndarray): Validation labels.
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    gd_iterations = [100, 500, 1000, 10000]
    
    results = []
    for lr in learning_rates:
        for n_iter in gd_iterations:
            theta_opt = grad_descent(X_train, y_train, lr, n_iter)
            train_error = classif_error(y_train, log_regr(X_train, theta_opt))
            val_error = classif_error(y_val, log_regr(X_val, theta_opt))
            results.append((lr, n_iter, train_error, val_error))
            print(f"Learning Rate: {lr}, Iterations: {n_iter}, Train Error: {train_error:.2f}%, Val Error: {val_error:.2f}%")

    # Display results as a table
    print("\nHyperparameter Tuning Results:")
    print("Learning Rate | Iterations | Train Error (%) | Val Error (%)")
    for res in results:
        print(f"{res[0]:<14} | {res[1]:<10} | {res[2]:<15.2f} | {res[3]:<12.2f}")

    # Plot train and validation errors
    plt.figure(figsize=(10, 6))
    lr_niter_labels = [f"lr={res[0]}, iters={res[1]}" for res in results]
    train_errors = [res[2] for res in results]
    val_errors = [res[3] for res in results]

    plt.plot(lr_niter_labels, train_errors, label='Train Error', marker='o')
    plt.plot(lr_niter_labels, val_errors, label='Validation Error', marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Error (%)')
    plt.title('Hyperparameter Tuning: Train vs Validation Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""

# New implementation of hyperparameter_tuning function
import numpy as np
import matplotlib.pyplot as plt
from models.logistic_regression import create_features_for_poly, log_regr
from models.optimizer import grad_descent
from scripts.utils import classif_error

def hyperparameter_tuning(X_train, y_train, X_val, y_val, degrees=[1, 2, 3], learning_rates=[0.01, 0.1, 0.5],
                          iterations=[500, 1000], random_init=True):
    """
    Perform hyperparameter tuning, including learning rates, iterations, and polynomial degrees.

    Args:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.
        degrees (list): List of polynomial degrees to test.
        learning_rates (list): List of learning rates to test.
        iterations (list): List of iteration counts to test.
        random_init (bool): If True, initialize theta randomly.

    Returns:
        None. Displays results and plots.
    """
    results = []

    # Iterate over all combinations of hyperparameters
    for degree in degrees:
        X_train_poly = create_features_for_poly(X_train, degree)
        X_train_poly = np.c_[X_train_poly, np.ones(X_train_poly.shape[0])]  # Add bias term

        X_val_poly = create_features_for_poly(X_val, degree)
        X_val_poly = np.c_[X_val_poly, np.ones(X_val_poly.shape[0])]

        for lr in learning_rates:
            for n_iter in iterations:
                theta = grad_descent(X_train_poly, y_train, learning_rate=lr, iters_total=n_iter, random_init=random_init)

                train_error = classif_error(y_train, log_regr(X_train_poly, theta))
                val_error = classif_error(y_val, log_regr(X_val_poly, theta))

                results.append((degree, lr, n_iter, train_error, val_error))
                print(f"Degree: {degree}, LR: {lr}, Iters: {n_iter}, Train Error: {train_error:.2f}%, Val Error: {val_error:.2f}%")

    # Plot results
    degrees_labels = [f"d={res[0]}, lr={res[1]}, iter={res[2]}" for res in results]
    train_errors = [res[3] for res in results]
    val_errors = [res[4] for res in results]

    plt.figure(figsize=(12, 6))
    plt.plot(degrees_labels, train_errors, label='Train Error', marker='o')
    plt.plot(degrees_labels, val_errors, label='Validation Error', marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Polynomial Degree, Learning Rate, Iterations")
    plt.ylabel("Error (%)")
    plt.title("Hyperparameter Tuning Results")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
