import joblib
import numpy as np
from sklearn.datasets import load_wine
from NN import normal_scaler

# Load neural net
nn = joblib.load("wineNN.joblib")        

data = load_wine()
X = data.data
y = data.target

# Fit scaler with standard deviation and mean from training dataset
scaler = normal_scaler()
scaler.s = [0.817222973, 1.13989881, 0.278232583, 3.44260619, 14.5991147, 0.635465788, 1.00062785, 0.127816106, 0.581597169, 2.32269541, 0.233275142, 0.719593358, 301.257195]
scaler.mu = [12.9790845, 2.37352113, 2.36084507, 19.4732394, 100.443662, 2.28908451, 2.00211268, 0.368028169, 1.60802817, 5.05760563, 0.956380282, 2.59281690, 734.894366]
X_hat = scaler.use_trained(X = X)

# Predictions
preds = nn.predict(X_hat)
print(preds)