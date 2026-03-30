import nlp
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('data_for_preprocessing.csv', header=0, index_col=0)
X = list(df['Text'])
y = list(df['Author'])

authors = sorted(set(y))
label2idx = {a: i for i, a in enumerate(authors)}
y = np.array([label2idx[a] for a in y])

np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(len(X) * 0.8)
X_train = [X[i] for i in indices[:split]]
X_test  = [X[i] for i in indices[split:]]
y_train = y[indices[:split]]
y_test  = y[indices[split:]]

if __name__ == "__main__":
    MAX_LEN = 100

    tokeniser = nlp.Tokeniser(max_vocab=1000)
    tokeniser.fit(X_train)

    X_train_enc = tokeniser.encode_batch(X_train, max_len=MAX_LEN)
    X_test_enc  = tokeniser.encode_batch(X_test,  max_len=MAX_LEN)

    embed_dim   = 16
    num_filters = 32
    num_classes = len(authors)

    model = nlp.NLPNetwork([
        nlp.Embedding(tokeniser.vocab_size, embed_dim),
        nlp.Conv1D(embed_dim, num_filters, kernel_size=3),
        nlp.ReLU(),
        nlp.GlobalMaxPool(),
        nlp.Dense(num_filters, 16),
        nlp.ReLU(),
        nlp.Dropout(0.3),
        nlp.Dense(16, num_classes),
        nlp.Softmax(),
    ])

#    model = nlp.NLPNetwork([
#        nlp.Embedding(tokeniser.vocab_size, embed_dim),
#        nlp.ParallelConv1D(embed_dim, num_filters, kernel_sizes),
#        nlp.ReLU(),
#        nlp.Dense(num_filters * len(kernel_sizes), 64),
#        nlp.ReLU(),
#        nlp.Dropout(0.3),
#        nlp.Dense(64, num_classes),
#        nlp.Softmax(),
#    ])

    model.train(X_train_enc, y_train, epochs=10, lr=0.005, batch_size=16)

    preds = model.predict(X_test_enc)
    acc = np.mean(preds == y_test)
    print(f"Test accuracy: {acc:.4f}")
    print(f"{int((acc)*len(preds))} correct out of {len(preds)}")
    print(f"{int((1-acc)*len(preds))} wrong out of {len(preds)}")
    joblib.dump(model, "nlptest.joblib")
    joblib.dump(tokeniser, "nlptesttokeniser.joblib")