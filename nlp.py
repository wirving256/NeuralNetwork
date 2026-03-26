import numpy as np
from collections import Counter


# ===================================================================
# Tokenizer


class Tokeniser:
    def __init__(self, max_vocab=10000, min_freq=1):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(text.lower().split())

        # Reserve 0=<PAD>, 1=<UNK>
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counts.most_common(self.max_vocab - 2):
            if freq < self.min_freq:
                break
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.lower().split()]

    def decode(self, indices):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in indices)

    def encode_batch(self, texts, max_len=None):
        encoded = [self.encode(t) for t in texts]
        if max_len is None:
            max_len = max(len(e) for e in encoded)
        # Pad or truncate
        out = np.zeros((len(encoded), max_len), dtype=np.int32)
        for i, e in enumerate(encoded):
            length = min(len(e), max_len)
            out[i, :length] = e[:length]
        return out


# ===================================================================
# Embedding Layer


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        # Initialise small random embeddings
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
        self.dW = np.zeros_like(self.W)
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)

    def forward(self, X):
        # X: (batch, seq_len) integer indices
        self.X = X
        return self.W[X]   # (batch, seq_len, embed_dim)

    def backward(self, dOut):
        # dOut: (batch, seq_len, embed_dim)
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.X, dOut)

    def update(self, lr, beta1, beta2, eps, t):
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        mW_hat = self.mW / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)


# ===================================================================
# Bag of Words pooling  (mean over sequence dimension)


class MeanPool:
    def forward(self, X, mask=None):
        # X: (batch, seq_len, embed_dim)
        # mask: (batch, seq_len) 1=real token, 0=pad
        self.X = X
        if mask is not None:
            self.mask = mask[:, :, np.newaxis]          # (batch, seq_len, 1)
            lengths = mask.sum(axis=1, keepdims=True)   # (batch, 1)
            self.lengths = np.maximum(lengths, 1)[:, :, np.newaxis]
            return (X * self.mask).sum(axis=1) / self.lengths.squeeze(-1)
        self.mask = None
        self.lengths = X.shape[1]
        return X.mean(axis=1)   # (batch, embed_dim)

    def backward(self, dOut):
        # dOut: (batch, embed_dim)
        batch, seq_len, embed_dim = self.X.shape
        dOut_expanded = dOut[:, np.newaxis, :] / (
            self.lengths if self.mask is not None else seq_len
        )
        if self.mask is not None:
            return dOut_expanded * self.mask
        return np.broadcast_to(dOut_expanded, self.X.shape).copy()


# ===================================================================
# 1D Convolutional layer  (text CNN — Kim 2014 style)


class Conv1D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.W = np.random.randn(out_channels, kernel_size, in_channels) * scale
        self.b = np.zeros((1, out_channels))
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, X):
        # X: (batch, seq_len, in_channels)
        self.X = X
        batch, seq_len, _ = X.shape
        out_len = seq_len - self.kernel_size + 1
        out = np.zeros((batch, out_len, self.out_channels))
        for k in range(self.kernel_size):
            out += X[:, k:k + out_len, :] @ self.W[:, k, :].T
        out += self.b
        return out   # (batch, out_len, out_channels)

    def backward(self, dOut):
        # dOut: (batch, out_len, out_channels)
        batch, seq_len, _ = self.X.shape
        out_len = dOut.shape[1]
        self.dW = np.zeros_like(self.W)
        self.db = dOut.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        dX = np.zeros_like(self.X)
        for k in range(self.kernel_size):
            # dW[:, k, :] += X[:, k:k+out_len, :].T @ dOut  (summed over batch)
            self.dW[:, k, :] += (self.X[:, k:k + out_len, :].transpose(0, 2, 1) @ dOut).sum(0).T
            dX[:, k:k + out_len, :] += dOut @ self.W[:, k, :]
        return dX

    def update(self, lr, beta1, beta2, eps, t):
        for param, grad, m, v in [
            (self.W, self.dW, self.mW, self.vW),
            (self.b, self.db, self.mb, self.vb),
        ]:
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ===================================================================
# 1D Parallel Convolutional layer

class ParallelConv1D:
    def __init__(self, in_channels, out_channels, kernel_sizes):
        self.convs = [Conv1D(in_channels, out_channels, k) for k in kernel_sizes]
        self.pools = [GlobalMaxPool() for _ in kernel_sizes]
        self._last_out = None

    def forward(self, X):
        self.X = X
        outs = []
        for conv, pool in zip(self.convs, self.pools):
            z = conv.forward(X)
            outs.append(pool.forward(z))
        self._last_out = outs
        return np.concatenate(outs, axis=1)  # (batch, out_channels * num_kernels)

    def backward(self, dOut):
        out_channels = dOut.shape[1] // len(self.convs)
        dX = np.zeros_like(self.X)
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            d_slice = dOut[:, i * out_channels:(i + 1) * out_channels]
            d_pool  = pool.backward(d_slice)
            dX     += conv.backward(d_pool)
        return dX

    def update(self, lr, beta1, beta2, eps, t):
        for conv in self.convs:
            conv.update(lr, beta1, beta2, eps, t)

# ===================================================================
# Global Max Pooling over sequence


class GlobalMaxPool:
    def forward(self, X):
        # X: (batch, seq_len, channels)
        self.X = X
        self.argmax = X.argmax(axis=1)          # (batch, channels)
        return X.max(axis=1)                     # (batch, channels)

    def backward(self, dOut):
        batch, seq_len, channels = self.X.shape
        dX = np.zeros_like(self.X)
        b_idx = np.arange(batch)[:, np.newaxis]
        c_idx = np.arange(channels)[np.newaxis, :]
        dX[b_idx, self.argmax, c_idx] = dOut
        return dX


# ===================================================================
# ReLU activation


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dX):
        return dX * self.mask


# ===================================================================
# Sigmoid activation


class Sigmoid:
    def forward(self, X):
        self.A = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        return self.A

    def backward(self, dX):
        return dX * self.A * (1 - self.A)


# ===================================================================
# Dense layer with Adam 


class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
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
        self.dW = self.X.T @ dZ / m
        self.db = dZ.sum(axis=0, keepdims=True) / m
        return dZ @ self.W.T

    def update(self, lr, beta1, beta2, eps, t):
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2
        mW_hat = self.mW / (1 - beta1 ** t)
        mb_hat = self.mb / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ===================================================================
# Dropout


class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.rate) / (1 - self.rate)
            return X * self.mask
        return X

    def backward(self, dX):
        return dX * self.mask


# ===================================================================
# Softmax + Cross Entropy Loss 


class Softmax:
    def forward(self, Z):
        e = np.exp(Z - Z.max(axis=1, keepdims=True))
        self.A = e / e.sum(axis=1, keepdims=True)
        return self.A

    def backward(self, dA):
        return dA


def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(np.log(y_pred[np.arange(m), y_true]))


def cross_entropy_grad(y_pred, y_true):
    m = y_pred.shape[0]
    grad = y_pred.copy()
    grad[np.arange(m), y_true] -= 1
    return grad / m


# ===================================================================
# NLP Network  (supports Embedding, Conv1D, Dense, Dropout, etc.)


class NLPNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.t = 0

    def forward(self, X, training=True):
        out = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                out = layer.forward(out, training=training)
            elif isinstance(layer, MeanPool):
                out = layer.forward(out)
            else:
                out = layer.forward(out)
        return out

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)

    def update(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr, beta1, beta2, eps, self.t)

    def train(self, X, y, epochs=10, lr=0.001, batch_size=32):
        n = len(X)
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X[:] = X[indices]
            y[:] = y[indices]

            epoch_loss = 0.0
            batches = 0
            for i in range(0, n, batch_size):
                Xb = X[i:i + batch_size]
                yb = y[i:i + batch_size]

                y_pred = self.forward(Xb, training=True)
                epoch_loss += cross_entropy_loss(y_pred, yb)

                dLoss = cross_entropy_grad(y_pred, yb)
                self.backward(dLoss)
                self.update(lr)
                batches += 1

            if epoch % 1 == 0:
                print(f"Epoch {epoch + 1}, Loss: {epoch_loss / batches:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)


# ===================================================================
# TF-IDF Vectorizer


class TFIDFVectorizer:
    def __init__(self, max_vocab=10000, min_freq=1):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.word2idx = {}
        self.idf = None

    def fit(self, texts):
        n = len(texts)
        df = Counter()
        tf_counts = Counter()
        for text in texts:
            words = set(text.lower().split())
            df.update(words)
            tf_counts.update(text.lower().split())

        vocab = [w for w, c in tf_counts.most_common(self.max_vocab)
                 if df[w] >= self.min_freq]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        # Smooth IDF: log((1+n)/(1+df)) + 1
        self.idf = np.array([
            np.log((1 + n) / (1 + df[w])) + 1 for w in vocab
        ])

    def transform(self, texts):
        n = len(texts)
        V = len(self.word2idx)
        X = np.zeros((n, V))
        for i, text in enumerate(texts):
            words = text.lower().split()
            for w in words:
                if w in self.word2idx:
                    X[i, self.word2idx[w]] += 1
            # TF normalisation
            total = X[i].sum()
            if total > 0:
                X[i] /= total
        return X * self.idf   # TF-IDF

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


# ===================================================================
# Word2Vec  (Skip-gram with negative sampling)


class Word2Vec:
    def __init__(self, vocab_size, embed_dim=50, neg_samples=5, lr=0.025):
        self.embed_dim = embed_dim
        self.neg_samples = neg_samples
        self.lr = lr
        # Target and context embedding matrices
        self.W_in  = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def train_pair(self, center, context, negatives):
        # Positive sample
        h = self.W_in[center]                         # (embed_dim,)
        pos_score = self._sigmoid(h @ self.W_out[context])
        pos_grad_out = (pos_score - 1) * h
        pos_grad_in  = (pos_score - 1) * self.W_out[context]

        # Negative samples
        neg_scores = self._sigmoid(self.W_out[negatives] @ h)  # (neg_samples,)
        neg_grad_out = neg_scores[:, np.newaxis] * h            # (neg_samples, embed_dim)
        neg_grad_in  = (neg_scores @ self.W_out[negatives])     # (embed_dim,)

        # Update
        self.W_out[context]   -= self.lr * pos_grad_out
        self.W_out[negatives] -= self.lr * neg_grad_out
        self.W_in[center]     -= self.lr * (pos_grad_in + neg_grad_in)

        # Loss (for monitoring)
        loss = -np.log(pos_score + 1e-12) - np.log(1 - neg_scores + 1e-12).sum()
        return loss

    def train(self, token_ids, window=2, epochs=3):
        vocab_size = self.W_in.shape[0]
        for epoch in range(epochs):
            total_loss = 0.0
            pairs = 0
            for i, center in enumerate(token_ids):
                lo = max(0, i - window)
                hi = min(len(token_ids), i + window + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    context = token_ids[j]
                    negatives = np.random.randint(0, vocab_size, self.neg_samples)
                    total_loss += self.train_pair(center, context, negatives)
                    pairs += 1
            print(f"Epoch {epoch + 1}, Loss: {total_loss / max(pairs, 1):.4f}")

    def get_embedding(self, idx):
        return self.W_in[idx]

    def most_similar(self, idx, top_k=5):
        vec = self.W_in[idx]
        # Cosine similarity
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-12
        sims = self.W_in @ vec / (norms.squeeze() * (np.linalg.norm(vec) + 1e-12))
        top = np.argsort(sims)[::-1][1:top_k + 1]
        return [(i, sims[i]) for i in top]



# ===================================================================
# Examples

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Example 1: Text CNN Classifier")
    print("=" * 60)
    texts = [
        "i love this film it was great",
        "wonderful movie highly recommend",
        "amazing performance loved every moment",
        "fantastic story brilliant acting",
        "great plot and wonderful characters",
        "terrible movie waste of time",
        "awful film hated every minute",
        "boring and dull complete disaster",
        "horrible acting terrible story",
        "worst film i have ever seen",
    ] * 20   # repeat to give more training signal

    labels = np.array(([0] * 5 + [1] * 5) * 20)   # 0=positive, 1=negative

    tokenizer = Tokeniser(max_vocab=500)
    tokenizer.fit(texts)

    X = tokenizer.encode_batch(texts, max_len=10)   # (n, 10)

    embed_dim = 16
    num_filters = 32
    num_classes = 2

    model = NLPNetwork([
        Embedding(tokenizer.vocab_size, embed_dim),
        Conv1D(embed_dim, num_filters, kernel_size=3),
        ReLU(),
        GlobalMaxPool(),
        Dense(num_filters, 16),
        ReLU(),
        Dropout(0.3),
        Dense(16, num_classes),
        Softmax(),
    ])

    model.train(X, labels, epochs=10, lr=0.005, batch_size=16)

    # After training your model...

    new_texts = [
        "i love this film it was great",
        "boring and terrible waste of time"
    ]

    # Use the SAME tokenizer that was fit on training data
    X_new = tokenizer.encode_batch(new_texts, max_len=12)  # same max_len as training

    preds = model.predict(X_new)
    print(preds)  # e.g. [0, 1]
    preds = model.predict(X)
    acc = np.mean(preds == labels)
    print(f"Train accuracy: {acc:.4f}")


#===========================================================
    print("\n" + "=" * 60)
    print("Example 2: Embedding + Mean Pool Classifier")
    print("=" * 60)

    texts = [
        "the cat sat on the mat",
        "dogs are great pets",
        "i enjoy hiking in the mountains",
        "the dog ran across the field",
        "cats sleep a lot during the day",
        "mountains are beautiful in summer",
        "my pet dog loves to play fetch",
        "the cat chased the mouse around",
        "hiking trails are great exercise",
        "pets bring joy to families",
    ] * 15

    labels = np.array(([0, 0, 1, 0, 0, 1, 0, 0, 1, 0]) * 15)  # 1 = hiking topic

    tokenizer = Tokeniser(max_vocab=200)
    tokenizer.fit(texts)
    X = tokenizer.encode_batch(texts, max_len=12)

    model = NLPNetwork([
        Embedding(tokenizer.vocab_size, 16),
        MeanPool(),
        Dense(16, 32),
        ReLU(),
        Dense(32, 2),
        Softmax(),
    ])

    new_texts = [
        "neural networks hehehehehehehehehe",
        "in the mountains i go hiking"
    ]

    # Use the SAME tokenizer that was fit on training data
    X_new = tokenizer.encode_batch(new_texts, max_len=12)  # same max_len as training

    model.train(X, labels, epochs=10, lr=0.01, batch_size=16)

    preds = model.predict(X_new)
    print(preds)  # e.g. [0, 1]

    preds = model.predict(X)
    print(f"Train accuracy: {np.mean(preds == labels):.4f}")

    #================================================

    print("\n" + "=" * 60)
    print("Example 3: TF-IDF + Dense Classifier")
    print("=" * 60)

    texts = [
        "python is a great programming language",
        "machine learning with numpy and python",
        "deep learning neural networks backprop",
        "data science statistics regression",
        "football is a popular sport worldwide",
        "the team scored a great goal today",
        "basketball players train very hard",
        "sports injuries require careful recovery",
    ] * 15

    labels = np.array(([0] * 4 + [1] * 4) * 15)   # 0=tech, 1=sports

    tfidf = TFIDFVectorizer(max_vocab=200)
    X = tfidf.fit_transform(texts).astype(np.float32)

    input_dim = X.shape[1]
    model = NLPNetwork([
        Dense(input_dim, 32),
        ReLU(),
        Dense(32, 2),
        Softmax(),
    ])

    model.train(X, labels, epochs=10, lr=0.01, batch_size=16)

    texts = [
        "neural networks use backprop",
        "football players train hard"
    ]

    X_new = tfidf.transform(texts)
    preds = model.predict(X_new)
    print(preds)

    preds = model.predict(X)
    print(f"Train accuracy: {np.mean(preds == labels):.4f}")