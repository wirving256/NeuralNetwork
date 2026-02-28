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



# ==================================================================
# Activations


class ReLU:
    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask

    def backward(self, dA):
        return dA * self.mask


class Softmax:
    def forward(self, Z):
        exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.A

    def backward(self, dA):
        return dA  # handled with CE gradient


# ==================================================================
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

        self.dW = self.X.T @ dZ / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return dZ @ self.W.T

    def update(self, lr, beta1, beta2, eps, t):
        # Update biased first moment
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        # Update biased second moment
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        # Bias correction
        mW_hat = self.mW / (1 - beta1 ** t)
        mb_hat = self.mb / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)

        # Update parameters
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ===================================================================
# Loss (Cross Entropy with integer labels)


def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(np.log(y_pred[np.arange(m), y_true]))


def cross_entropy_grad(y_pred, y_true):
    m = y_pred.shape[0]
    grad = y_pred.copy()
    grad[np.arange(m), y_true] -= 1
    return grad / m


# =================================================================
# Neural Network


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.t = 0  # Adam timestep

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

                # Forward
                y_pred = self.forward(X_batch)

                # Backward
                dLoss = cross_entropy_grad(y_pred, y_batch)
                self.backward(dLoss)

                # Adam update
                self.update(lr)

            if epoch % 50 == 0:
                loss = cross_entropy_loss(self.forward(X), y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# ==================================================================
# Example: Iris Dataset


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X = data.data
    y = data.target  # integer labels (0,1,2)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nn = NeuralNetwork([
        Dense(4, 32),
        ReLU(),
        Dense(32, 16),
        ReLU(),
        Dense(16, 3),
        Softmax()
    ])

    nn.train(X_train, y_train, epochs=1000, lr=0.01)

    preds = nn.predict(X_test)
    acc = np.mean(preds == y_test)
    print("Test Accuracy:", acc)
    wrong = 0
    for i, index in enumerate(preds):
        if preds[i] != y_test[i]:
            wrong += 1
    print(f"{wrong} wrong out of {len(preds)}")