import numpy as np


# ================================================================
# Normal Scaler (for tabular use if needed)

class normal_scaler:
    def __init__(self):
        self.s = None
        self.mu = None

    def fit_params(self, X):
        self.mu = np.mean(X, axis=0)
        self.s = np.std(X, axis=0) + 1e-8

    def use_trained(self, X):
        return (X - self.mu) / self.s


# ================================================================
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
        return dA


# ================================================================
# im2col utilities (vectorized convolution)

def im2col(X, KH, KW, stride=1, padding=0):
    N, C, H, W = X.shape

    H_out = (H + 2*padding - KH)//stride + 1
    W_out = (W + 2*padding - KW)//stride + 1

    if padding > 0:
        X = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)))

    cols = np.zeros((N, C, KH, KW, H_out, W_out))

    for y in range(KH):
        y_max = y + stride*H_out
        for x in range(KW):
            x_max = x + stride*W_out
            cols[:, :, y, x, :, :] = X[:, :, y:y_max:stride, x:x_max:stride]

    cols = cols.transpose(0,4,5,1,2,3).reshape(N*H_out*W_out, -1)
    return cols


def col2im(cols, X_shape, KH, KW, stride=1, padding=0):
    N, C, H, W = X_shape

    H_out = (H + 2*padding - KH)//stride + 1
    W_out = (W + 2*padding - KW)//stride + 1

    cols = cols.reshape(N, H_out, W_out, C, KH, KW).transpose(0,3,4,5,1,2)

    X_padded = np.zeros((N, C, H + 2*padding, W + 2*padding))

    for y in range(KH):
        y_max = y + stride*H_out
        for x in range(KW):
            x_max = x + stride*W_out
            X_padded[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]

    if padding > 0:
        return X_padded[:, :, padding:-padding, padding:-padding]
    return X_padded


# ================================================================
# Fully Vectorized Conv2D

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.KH = kernel_size
        self.KW = kernel_size

        scale = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels,))

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        F = self.W.shape[0]

        self.cols = im2col(X, self.KH, self.KW, self.stride, self.padding)
        W_col = self.W.reshape(F, -1)

        out = self.cols @ W_col.T + self.b

        H_out = (H + 2*self.padding - self.KH)//self.stride + 1
        W_out = (W + 2*self.padding - self.KW)//self.stride + 1

        out = out.reshape(N, H_out, W_out, F).transpose(0,3,1,2)
        return out

    def backward(self, dZ):
        N, F, H_out, W_out = dZ.shape

        dZ_flat = dZ.transpose(0,2,3,1).reshape(-1, F)
        W_col = self.W.reshape(F, -1)

        self.dW = (dZ_flat.T @ self.cols).reshape(self.W.shape)
        self.db = np.sum(dZ_flat, axis=0)

        dcols = dZ_flat @ W_col
        dX = col2im(dcols, self.X.shape, self.KH, self.KW, self.stride, self.padding)

        return dX

    def update(self, lr, beta1, beta2, eps, t):
        self.mW = beta1*self.mW + (1-beta1)*self.dW
        self.vW = beta2*self.vW + (1-beta2)*(self.dW**2)

        self.mb = beta1*self.mb + (1-beta1)*self.db
        self.vb = beta2*self.vb + (1-beta2)*(self.db**2)

        mW_hat = self.mW/(1-beta1**t)
        vW_hat = self.vW/(1-beta2**t)

        mb_hat = self.mb/(1-beta1**t)
        vb_hat = self.vb/(1-beta2**t)

        self.W -= lr*mW_hat/(np.sqrt(vW_hat)+eps)
        self.b -= lr*mb_hat/(np.sqrt(vb_hat)+eps)


# ================================================================
# MaxPool2D

class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape

        H_out = H // self.size
        W_out = W // self.size

        self.out = X.reshape(N, C, H_out, self.size, W_out, self.size).max(axis=(3,5))
        return self.out

    def backward(self, dZ):
        N, C, H, W = self.X.shape
        H_out = H // self.size
        W_out = W // self.size

        dX = np.zeros_like(self.X)

        X_reshaped = self.X.reshape(N, C, H_out, self.size, W_out, self.size)
        max_mask = (X_reshaped == X_reshaped.max(axis=(3,5), keepdims=True))

        dZ_expanded = dZ[:, :, :, None, :, None]
        dX = (max_mask * dZ_expanded).reshape(self.X.shape)

        return dX


# ================================================================
# Flatten

class Flatten:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dZ):
        return dZ.reshape(self.shape)


# ================================================================
# Dense Layer (Adam)

class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))

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
        self.mW = beta1*self.mW + (1-beta1)*self.dW
        self.vW = beta2*self.vW + (1-beta2)*(self.dW**2)

        self.mb = beta1*self.mb + (1-beta1)*self.db
        self.vb = beta2*self.vb + (1-beta2)*(self.db**2)

        mW_hat = self.mW/(1-beta1**t)
        vW_hat = self.vW/(1-beta2**t)

        mb_hat = self.mb/(1-beta1**t)
        vb_hat = self.vb/(1-beta2**t)

        self.W -= lr*mW_hat/(np.sqrt(vW_hat)+eps)
        self.b -= lr*mb_hat/(np.sqrt(vb_hat)+eps)


# ================================================================
# Loss

def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1-1e-12)
    return -np.mean(np.log(y_pred[np.arange(m), y_true]))


def cross_entropy_grad(y_pred, y_true):
    m = y_pred.shape[0]
    grad = y_pred.copy()
    grad[np.arange(m), y_true] -= 1
    return grad / m


# ================================================================
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
            if hasattr(layer, "update"):
                layer.update(lr, beta1, beta2, eps, self.t)

    def train(self, X, y, epochs=5, lr=0.001, batch_size=64):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.forward(X_batch)
                dLoss = cross_entropy_grad(y_pred, y_batch)
                self.backward(dLoss)
                self.update(lr)

            loss = cross_entropy_loss(self.forward(X[:1000]), y[:1000])
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
if __name__ == "__main__":

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    # ============================================================
    # Load dataset
    digits = load_digits()

    X = digits.images            # shape (1797, 8, 8)
    y = digits.target            # integer labels 0–9

    # Normalize (pixel values are 0–16)
    X = X / 16.0

    # Reshape to CNN format (N, C, H, W)
    X = X.reshape(-1, 1, 8, 8)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================================================
    # Build CNN

    cnn = NeuralNetwork([
        Conv2D(1, 8, 3),        # (8x8 → 6x6)
        ReLU(),
        MaxPool2D(),            # (6x6 → 3x3)
        Flatten(),
        Dense(8*3*3, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ])

    # ============================================================
    # Train

    cnn.train(X_train, y_train, epochs=500, lr=0.001, batch_size=64)

    # ============================================================
    # Evaluate

    preds = cnn.predict(X_test)
    acc = np.mean(preds == y_test)

    print("\nTest Accuracy:", acc)
    print(f"{np.sum(preds != y_test)} wrong predictions out of {len(preds)}")