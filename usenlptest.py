import joblib
import numpy as np
import pandas as pd

df = pd.read_csv('data_for_preprocessing.csv', header=0, index_col=0)
X = list(df['Text'])
y = list(df['Author'])

authors = sorted(set(y))
label2idx = {a: i for i, a in enumerate(authors)}
y = np.array([label2idx[a] for a in y])

np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(len(X) * 0.8)
X_test  = [X[i] for i in indices[split:]]
y_test  = y[indices[split:]]

# Load neural net
model = joblib.load("nlptest.joblib")        

# Fit scaler with standard deviation and mean from training dataset
tokeniser = joblib.load("nlptesttokeniser.joblib")
X_test = tokeniser.encode_batch(X_test, max_len=100)

# Predictions
preds = model.predict(X_test)
acc = np.mean(preds == y_test)
print(f"test accuracy: {acc}")
print(f"{int((acc)*len(preds))} correct out of {len(preds)}")
print(f"{int((1-acc)*len(preds))} wrong out of {len(preds)}")