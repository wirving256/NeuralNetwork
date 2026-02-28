if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_wine
    import NN
    import joblib

    data = load_wine()
    X = data.data
    y = data.target  # integer labels (0,1,2)

    # Split to training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = NN.normal_scaler()
    scaler.fit_params(X_train)
    X_train = scaler.use_trained(X_train)

    # Set up NN
    nn = NN.NeuralNetwork([
        NN.Dense(13, 32),
        NN.ReLU(),
        NN.Dense(32, 32),
        NN.ReLU(),
        NN.Dense(32, 10),
        NN.Softmax()
    ])

    # Scale test set
    X_test = scaler.use_trained(X_test)
    s_scaler, mu_scaler = scaler.return_params()
    print(f"s = {s_scaler}, mu = {mu_scaler}")

    # Train model on training set
    nn.train(X_train, y_train, epochs=1000, lr=0.01)

    # Test accuracy on test set
    preds = nn.predict(X_test)
    acc = np.mean(preds == y_test)
    print("Test Accuracy:", acc)
    wrong = 0
    for i, index in enumerate(preds):
        if preds[i] != y_test[i]:
            wrong += 1
    print(f"{wrong} wrong out of {len(preds)}")

    # Dump model for future use
    joblib.dump(nn, "wineNN.joblib")