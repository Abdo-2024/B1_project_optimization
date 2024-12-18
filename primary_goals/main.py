from data_utils import create_data, plot_data
from logistic_regression import grad_descent, mean_logloss, log_regr, classif_error
from features import create_features_for_poly
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.01
    iters = 1000
    degree = 2
    n_samples_train, n_samples_val = 500, 1000

    # Generate and plot data
    X_train, class_labels_train = create_data(n_samples_train)
    plot_data(X_train, class_labels_train)

    # Prepare data
    y_train = (class_labels_train == 2).astype(int)
    X_train_poly = create_features_for_poly(X_train, degree)
    X_train_poly = np.c_[X_train_poly, np.ones(X_train_poly.shape[0])]  # Add bias term

    # Train model
    theta_opt = grad_descent(X_train_poly, y_train.reshape(-1, 1), learning_rate, iters)
    
    # Validate model
    X_val, class_labels_val = create_data(n_samples_val)
    y_val = (class_labels_val == 2).astype(int)
    X_val_poly = create_features_for_poly(X_val, degree)
    X_val_poly = np.c_[X_val_poly, np.ones(X_val_poly.shape[0])]

""" 
    Evaluate performance
    y_pred = (log_regr(X_val_poly, theta_opt) > 0.5).astype(int)
    error = classif_error(y_val, y_pred)
    print(f"Validation Error: {error:.2f}%")


    error = classif_error(y_val, y_pred)
    print(f"Validation Error: {error:.2f}%")
"""

def plot_decision_boundary(X, theta, degree):
    """
    Plots the decision boundary for logistic regression.
    Parameters:
        X (numpy.ndarray): Feature matrix (used for defining axis range).
        theta (numpy.ndarray): Model weights.
        degree (int): Polynomial degree for features.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_poly = create_features_for_poly(grid, degree)
    grid_poly = np.c_[grid_poly, np.ones(grid_poly.shape[0])]  # Add bias term

    probs = log_regr(grid_poly, theta).reshape(xx.shape)
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="RdYlGn", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=class_labels_train, edgecolors="k", cmap="RdYlGn")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()


    plot_decision_boundary(X_train, theta_opt, degree)

theta_opt, loss_history = grad_descent(X_train_poly, y_train.reshape(-1, 1), learning_rate, iters)
plt.plot(range(iters), loss_history)
plt.xlabel("Iterations")
plt.ylabel("Log-Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()


# Initialize variables to store results
degrees = range(1, 6)
training_losses = []
validation_errors = []

# Number of iterations for gradient descent
gd_iters = 1000

# Loop through different polynomial degrees
for degree in degrees:
    print(f"Training for Polynomial Degree: {degree}")
    
    # Polynomial Feature Transformation
    X_train_poly = create_features_for_poly(X_train, degree)
    X_train_poly = np.c_[X_train_poly, np.ones(X_train_poly.shape[0])]  # Add bias
    
    X_val_poly = create_features_for_poly(X_val, degree)
    X_val_poly = np.c_[X_val_poly, np.ones(X_val_poly.shape[0])]  # Add bias
    
    # Train the Model
    theta_opt, loss_history = grad_descent(X_train_poly, y_train.reshape(-1, 1), learning_rate, gd_iters)
    
    # Compute Training Loss
    train_loss = mean_logloss(X_train_poly, y_train.reshape(-1, 1), theta_opt)
    training_losses.append(train_loss)
    
    # Compute Validation Predictions
    y_val_pred = log_regr(X_val_poly, theta_opt) >= 0.5
    val_error = classif_error(y_val, y_val_pred)
    validation_errors.append(val_error)
    
    print(f"Degree {degree}: Training Loss = {train_loss:.4f}, Validation Error = {val_error:.2f}%")

# Create a Table
results_df = pd.DataFrame({
    "Degree": degrees,
    "Training Loss": training_losses,
    "Validation Error (%)": validation_errors
})
print("\nPerformance Results Across Degrees:\n", results_df)

# Plot Results
plt.figure(figsize=(12, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(degrees, training_losses, marker='o', label="Training Loss")
plt.xlabel("Polynomial Degree")
plt.ylabel("Log-Loss")
plt.title("Training Loss vs. Polynomial Degree")
plt.grid(True)
plt.legend()

# Plot Validation Error
plt.subplot(1, 2, 2)
plt.plot(degrees, validation_errors, marker='o', color="red", label="Validation Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Error (%)")
plt.title("Validation Error vs. Polynomial Degree")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()