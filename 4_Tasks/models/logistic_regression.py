import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, degree_poly=1):
        """
        Logistic Regression Model.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations.
            degree_poly (int): Degree of polynomial features to use (1 = linear).
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.degree_poly = degree_poly
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def add_polynomial_features(self, X):
        """
        Generate polynomial features up to degree `degree_poly`.

        Args:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Transformed feature matrix with polynomial features.
        """
        if self.degree_poly == 1:
            return X
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(self.degree_poly, include_bias=False)
        return poly.fit_transform(X)

    def initialize_parameters(self, n_features):
        """
        Initialize weights and bias.

        Args:
            n_features (int): Number of features in the dataset.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def compute_loss(self, y_true, y_pred):
        """
        Compute log-loss.

        Args:
            y_true (numpy.ndarray): True labels.
            y_pred (numpy.ndarray): Predicted probabilities.

        Returns:
            float: Log-loss value.
        """
        m = y_true.shape[0]
        loss = -(1 / m) * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss

    def gradient_descent_step(self, X, y_true, y_pred):
        """
        Perform a single step of gradient descent.

        Args:
            X (numpy.ndarray): Input features.
            y_true (numpy.ndarray): True labels.
            y_pred (numpy.ndarray): Predicted probabilities.
        """
        m = X.shape[0]
        dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
        db = (1 / m) * np.sum(y_pred - y_true)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Args:
            X (numpy.ndarray): Training feature matrix.
            y (numpy.ndarray): Training labels.
        """
        X = self.add_polynomial_features(X)
        self.initialize_parameters(X.shape[1])

        for i in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            self.gradient_descent_step(X, y, y_pred)

            if i % 100 == 0:  # Log progress every 100 iterations
                loss = self.compute_loss(y, y_pred)
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Args:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        X = self.add_polynomial_features(X)
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predict binary labels (0 or 1) for input data.

        Args:
            X (numpy.ndarray): Input feature matrix.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

# Wrapper functions

def train_logistic_regression(X, y, learning_rate, max_iter, degree_poly=1):
    """
    Train a logistic regression model using the provided data.

    Args:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Training labels.
        learning_rate (float): Learning rate for gradient descent.
        max_iter (int): Maximum number of iterations.
        degree_poly (int): Degree of polynomial features to use.

    Returns:
        LogisticRegression: Trained logistic regression model.
    """
    model = LogisticRegression(learning_rate=learning_rate, max_iter=max_iter, degree_poly=degree_poly)
    model.fit(X, y)
    return model

def predict_logistic_regression(X, model):
    """
    Predict binary labels using a trained logistic regression model.

    Args:
        X (numpy.ndarray): Input feature matrix.
        model (LogisticRegression): Trained logistic regression model.

    Returns:
        numpy.ndarray: Predicted labels.
    """
    return model.predict(X)
