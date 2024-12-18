###############################################
# Main Orchestration Script for 4 Task
# Author: AbdoAllah Mohammad
# Description: Combines all modules to complete tasks from pages 7-10 of the B1 project.
###############################################

# Import dependencies
import numpy as np
from data.create_data import create_data
from models.gradient_descent import optimize_with_gradient_descent
from models.logistic_regression import train_logistic_regression, predict_logistic_regression
from main_project_skeleton import grad_descent, mean_logloss, log_regr, classif_error, plot_data
from utils.utilities import (
    create_features_for_poly,
    evaluate_loss,
    classif_error,
    plot_decision_boundary,  # Remove plot_data import
)


# Hyperparameters
learning_rate = 0.001
num_iterations = 1000
degree_poly = 2  # Polynomial degree for feature expansion
n_samples_train = 400
n_samples_val = 4000

# ---- Main Workflow ----
if __name__ == "__main__":
    print("Starting main pipeline...")

    # Step 1: Generate Data
    print("Generating training and validation data...")
    X_train, class_labels_train = create_data(n_samples_train)
    y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1

    X_val, class_labels_val = create_data(n_samples_val)
    y_val = (class_labels_val == 1) * 0 + (class_labels_val == 2) * 1

    # Step 2: Polynomial Feature Expansion
    print(f"Expanding features to degree {degree_poly}...")
    X_train_poly = create_features_for_poly(X_train, degree_poly)
    X_train_poly = np.concatenate((X_train_poly, np.ones((n_samples_train, 1))), axis=1)

    X_val_poly = create_features_for_poly(X_val, degree_poly)
    X_val_poly = np.concatenate((X_val_poly, np.ones((n_samples_val, 1))), axis=1)

    # Step 3: Train Logistic Regression Model
    print("Training logistic regression model using gradient descent...")
    theta_opt = train_logistic_regression(
        X_train_poly, y_train, learning_rate, num_iterations
    )

    # Step 4: Evaluate the Model
    print("Evaluating model performance...")
    y_pred_train = predict_logistic_regression(X_train_poly, theta_opt)
    y_pred_val = predict_logistic_regression(X_val_poly, theta_opt)

    loss_train = evaluate_loss(X_train_poly, y_train, theta_opt)
    loss_val = evaluate_loss(X_val_poly, y_val, theta_opt)

    error_train = evaluate_loss(y_train, y_pred_train)
    error_val = evaluate_loss(y_val, y_pred_val)

    print(f"Training Loss: {loss_train:.4f}, Validation Loss: {loss_val:.4f}")
    print(f"Training Error: {error_train:.2f}%, Validation Error: {error_val:.2f}%")

    # Step 5: Plot Data and Decision Boundary
    print("Plotting data and decision boundaries...")
    plot_data(X_train, class_labels_train)
    plot_decision_boundary(X_train, class_labels_train, theta_opt, degree_poly)

    plot_data(X_val, class_labels_val)
    plot_decision_boundary(X_val, class_labels_val, theta_opt, degree_poly)

    print("Pipeline complete!")
