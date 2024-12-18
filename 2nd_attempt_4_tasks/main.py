###############################################
# Author & Copyright: Konstantinos Kamnitsas
# Modified by: AbdoAllah Mohammad
# B1 - Project - 2024
###############################################
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt

from data.create_data import create_data
from models.logistic_regression import create_features_for_poly, log_regr
from models.optimizer import grad_descent
from scripts.utils import classif_error, mean_logloss
from scripts.visualization import plot_3d_sigmoid_surface, plot_decision_boundary
from experiments.hyperparameter_tuning import hyperparameter_tuning
from experiments.polynomial_analysis import polynomial_degree_comparison, training_size_analysis

if __name__ == "__main__":
    """
    Main execution script for:
        1. Hyperparameter Tuning
        2. Polynomial Degree Comparison
        3. Decision Boundary Visualization
    """

    # --- Hyperparameters ---
    max_degree = 5        # Maximum polynomial degree to test
    learning_rate = 0.1   # Default learning rate
    gd_iters = 1000       # Default gradient descent iterations

    # --- Create Training Data ---
    n_samples_train = 400
    X_train, class_labels_train = create_data(n_samples_train)
    y_train = (class_labels_train == 1).astype(int)  # Convert labels to {0, 1}

    # --- Create Validation Data ---
    n_samples_val = 4000
    X_val, class_labels_val = create_data(n_samples_val)
    y_val = (class_labels_val == 1).astype(int)  # Convert labels to {0, 1}

    # --- 1. Hyperparameter Tuning ---
    print("\n--- Hyperparameter Tuning ---")
    hyperparameter_tuning(X_train, y_train, X_val, y_val)

    # --- 2. Polynomial Degree Comparison ---
    print("\n--- Polynomial Degree Comparison ---")
    best_degree = polynomial_degree_comparison(X_train, y_train, X_val, y_val, max_degree)

    # --- 3. Retrain Model with Best Degree ---
    print(f"\nBest Polynomial Degree: {best_degree}")
    X_train_best = create_features_for_poly(X_train, best_degree)
    X_train_best = np.concatenate((X_train_best, np.ones((X_train_best.shape[0], 1))), axis=1)  # Add bias

    X_val_best = create_features_for_poly(X_val, best_degree)
    X_val_best = np.concatenate((X_val_best, np.ones((X_val_best.shape[0], 1))), axis=1)  # Add bias

    theta_opt = grad_descent(X_train_best, y_train, learning_rate, gd_iters)

    # --- 4. Evaluate Model ---
    train_loss = mean_logloss(X_train_best, y_train, theta_opt)
    val_loss = mean_logloss(X_val_best, y_val, theta_opt)
    train_error = classif_error(y_train, log_regr(X_train_best, theta_opt))
    val_error = classif_error(y_val, log_regr(X_val_best, theta_opt))

    print("\n--- Final Model Performance ---")
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Training Error: {train_error:.2f}%, Validation Error: {val_error:.2f}%")

    # --- 5. Analyze Training Sample Size Impact ---
    print("\n--- Training Sample Size Analysis ---")
    training_size_analysis(X_train, y_train, X_val, y_val, degree=best_degree, 
                           learning_rate=learning_rate, gd_iters=gd_iters)

    # --- 6. Plot Decision Boundaries ---
    plt.figure(figsize=(10, 5))

################################################# Extension #################################################
    # --- 7. 3D Sigmoid Surface Visualization ---
    print("\n--- 3D Sigmoid Surface Visualization ---")
    plot_3d_sigmoid_surface(X_train, class_labels_train, theta_opt, degree=best_degree)

    # Training Data
    plt.subplot(1, 2, 1)
    plot_decision_boundary(X_train, theta_opt, class_labels_train, best_degree, marker_size=8)
    plt.title("Training Data with Decision Boundary")

    # Validation Data
    plt.subplot(1, 2, 2)
    plot_decision_boundary(X_val, theta_opt, class_labels_val, best_degree, marker_size=8)
    plt.title("Validation Data with Decision Boundary")

    plt.tight_layout()
    plt.show()