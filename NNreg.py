import numpy as np


# ===================================================================
# Standard Normal Distribution Scaler


class normal_scaler:
    def __init__(self):
        self.s = None
        self.mu = None

    def fit_params(self, X):
        if not isinstance(X,np.ndarray):
            raise TypeError("Expected type np.ndarray")
        self.mu = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)
    
    def use_trained(self, X):
        return (X-self.mu)/self.s
    
    def return_params(self):
        return self.s, self.mu


# =====================================================================
# Activations


class ReLU:
    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask

    def backward(self, dA):
        return dA * self.mask


# ======================================================================
# Dense Layer with Adam


class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))

        # Adam parameters
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        m = self.X.shape[0]
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T

    def update(self, lr, beta1, beta2, eps, t):
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        mW_hat = self.mW / (1 - beta1 ** t)
        mb_hat = self.mb / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ====================================================================
# MSE Loss


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_grad(y_pred, y_true):
    m = y_true.shape[0]
    return 2 * (y_pred - y_true) / m


# =====================================================================
# Neural Network


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.t = 0

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)

    def update(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update(lr, beta1, beta2, eps, self.t)

    def train(self, X, y, epochs=500, lr=0.001, batch_size=32):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.forward(X_batch)

                dLoss = mse_grad(y_pred, y_batch)
                self.backward(dLoss)
                self.update(lr)

            if epoch % 50 == 0:
                loss = mse_loss(self.forward(X), y)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


# =================================================================
# Example: Simple Regression


if __name__ == "__main__":

    # Synthetic regression dataset
    np.random.seed(42)
    X = np.random.randn(500, 3)

    # True function: y = 3x1 - 2x2 + 0.5x3 + noise
    y = 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + 0.1*np.random.randn(500)

    y = y.reshape(-1, 1)  # Important: shape (n_samples, 1)

    nn = NeuralNetwork([
        Dense(3, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 1)   # Single regression output
    ])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    nn.train(X_train, y_train, epochs=500, lr=0.001)

    preds = nn.predict(X_test)
    print("Final MSE:", mse_loss(preds, y_test))
