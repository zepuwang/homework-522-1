import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = 0
        self.b = 0

    def fit(self, X, y) -> None:
        n = X.shape[0]
        p = X.shape[1]
        a = np.array([1] * n).reshape(n, 1)
        x_ = np.concatenate((X, a), axis=1)
        w_ = np.linalg.inv(x_.T @ x_) @ (x_.T @ y)
        self.w = w_[:-1]
        self.b = w_[-1]
        # raise NotImplementedError()

    def predict(self, X) -> np.ndarray:
        y = self.w @ X.T + self.b
        # raise NotImplementedError()
        return y


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # raise NotImplementedError()
        self.w = 0
        self.b = 0
        n = len(X)
        for i in range(epochs):
            y_pred = self.w * X + self.b
            dw = (-2 / n) * sum(X * (y - y_pred))
            db = (-1 / n) * sum(X * (y - y_pred))
            self.w = self.w - dw * lr
            self.b = self.b - db * lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        y_pred = self.w @ X.T + self.b
        return y_pred
        # raise NotImplementedError()
