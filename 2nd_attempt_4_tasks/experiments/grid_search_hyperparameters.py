import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
from models.logistic_regression import create_features_for_poly, log_regr
from models.optimizer import grad_descent
from scripts.utils import classif_error

def grid_search_hyperparameters(X_train, y_train, X_val, y_val, degrees, learning_rates, iterations):
    results = []

    # Iterate over degrees, learning rates, and iteration counts
    for degree in degrees:
        print(f"Testing Polynomial Degree {degree}...")
        X_train_poly = create_features_for_poly(X_train, degree)
        X_train_poly = np.c_[X_train_poly, np.ones(X_train_poly.shape[0])]  # Add bias
        X_val_poly = create_features_for_poly(X_val, degree)
        X_val_poly = np.c_[X_val_poly, np.ones(X_val_poly.shape[0])]  # Add bias

        for lr in learning_rates:
            for iters in iterations:
                # Perform gradient descent
                theta = grad_descent(X_train_poly, y_train, learning_rate=lr, iters_total=iters)
                
                # Calculate training and validation errors
                train_error = classif_error(y_train, log_regr(X_train_poly, theta))
                val_error = classif_error(y_val, log_regr(X_val_poly, theta))
                
                # Store results
                results.append((degree, lr, iters, train_error, val_error))
                print(f"Degree: {degree}, LR: {lr:.2f}, Iters: {iters} -> Train Error: {train_error:.2f}%, Val Error: {val_error:.2f}%")

    # Return all results
    return results

# usage
if __name__ == "__main__":
    # Hyperparameter ranges
    degrees = [1, 2, 3, 4, 5]
    learning_rates = np.arange(0.05, 0.55, 0.05)  # 0.05 to 0.5 in steps of 0.05
    iterations = range(100, 1100, 100)  # 100 to 1000 in steps of 100

    # Generate training and validation data
    from data.create_data import create_data

    n_samples_train = 400
    n_samples_val = 4000

    # Training data
    X_train, class_labels_train = create_data(n_samples_train)
    y_train = (class_labels_train == 1).astype(int)  # Convert labels to {0, 1}

    # Validation data
    X_val, class_labels_val = create_data(n_samples_val)
    y_val = (class_labels_val == 1).astype(int)  # Convert labels to {0, 1}

    # Perform grid search
    results = grid_search_hyperparameters(X_train, y_train, X_val, y_val, degrees, learning_rates, iterations)

    # Save results to a file
    with open("grid_search_results.txt", "w") as f:
        for res in results:
            f.write(f"Degree: {res[0]}, LR: {res[1]:.2f}, Iters: {res[2]}, Train Error: {res[3]:.2f}%, Val Error: {res[4]:.2f}%\n")
