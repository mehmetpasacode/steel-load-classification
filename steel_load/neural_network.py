"""NumPy implementation adapted from the original course project."""

import numpy as np

from .common import cross_entropy, softmax


def initialize_parameters(input_size, hidden_size, output_size, seed=42):
    rng = np.random.default_rng(seed)

    W1 = 0.01 * rng.standard_normal((input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))

    W2 = 0.01 * rng.standard_normal((hidden_size, output_size))
    b2 = np.zeros((1, output_size))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def tanh(Z):
    return np.tanh(Z)


def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def sigmoid_derivative(A):
    return A * (1 - A)


def tanh_derivative(A):
    return 1 - A**2


def leaky_relu_derivative(Z, alpha=0.01):
    grad = np.ones_like(Z)
    grad[Z < 0] = alpha
    return grad

def forward_propagation(X, parameters, hidden_activation="relu", output_activation="softmax"):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(X, W1) + b1

    if hidden_activation == "relu":
        A1 = relu(Z1)
    elif hidden_activation == "sigmoid":
        A1 = sigmoid(Z1)
    elif hidden_activation == "tanh":
        A1 = tanh(Z1)
    elif hidden_activation == "leaky_relu":
        A1 = leaky_relu(Z1)
    else:
        raise ValueError("Unknown hidden activation")

    Z2 = np.dot(A1, W2) + b2

    if output_activation == "softmax":
        Y_hat = softmax(Z2)
    elif output_activation == "sigmoid":
        Y_hat = sigmoid(Z2)
    elif output_activation == "linear":
        Y_hat = Z2
    else:
        raise ValueError("Unknown output activation")

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "Y_hat": Y_hat
    }

    return Y_hat, cache


def compute_loss(y_true, y_pred, loss_type="cross_entropy", class_weights=None):
    if loss_type == "cross_entropy":
        return cross_entropy(y_true, y_pred, class_weights)
    if loss_type == "mse":
        return np.mean((y_true - y_pred) ** 2)
    if loss_type == "mae":
        return np.mean(np.abs(y_true - y_pred))
    raise ValueError("Unknown loss type")


def backward_propagation(y_true, parameters, cache, hidden_activation="relu", loss_type="cross_entropy", class_weights=None):
    W2 = parameters["W2"]

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Y_hat = cache["Y_hat"]

    m = y_true.shape[0]
    C = y_true.shape[1]

    if loss_type == "cross_entropy":
        dZ2 = Y_hat - y_true

        if class_weights is not None:
            sample_weights = np.sum(y_true * class_weights.reshape(1, -1), axis=1, keepdims=True)
            dZ2 = dZ2 * sample_weights

    elif loss_type == "mse":
        dY = 2 * (Y_hat - y_true) / C
        dZ2 = Y_hat * (dY - np.sum(dY * Y_hat, axis=1, keepdims=True))

    elif loss_type == "mae":
        dY = np.sign(Y_hat - y_true) / C
        dZ2 = Y_hat * (dY - np.sum(dY * Y_hat, axis=1, keepdims=True))

    else:
        raise ValueError("Unknown loss type")

    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)

    if hidden_activation == "relu":
        dZ1 = dA1 * relu_derivative(Z1)
    elif hidden_activation == "sigmoid":
        dZ1 = dA1 * sigmoid_derivative(A1)
    elif hidden_activation == "tanh":
        dZ1 = dA1 * tanh_derivative(A1)
    elif hidden_activation == "leaky_relu":
        dZ1 = dA1 * leaky_relu_derivative(Z1)
    else:
        raise ValueError("Unknown hidden activation")

    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def initialize_velocity(parameters):
    velocity = {
        "dW1": np.zeros_like(parameters["W1"]),
        "db1": np.zeros_like(parameters["b1"]),
        "dW2": np.zeros_like(parameters["W2"]),
        "db2": np.zeros_like(parameters["b2"])
    }
    return velocity

def compute_class_weights(y_train):
    y_labels = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_labels, minlength=y_train.shape[1])
    if np.any(class_counts == 0):
        raise ValueError('Class weighting requires every class in the training split.')
    total_samples = len(y_labels)
    num_classes = len(class_counts)

    class_weights = total_samples / (num_classes * class_counts)
    return class_weights

def update_parameters_with_momentum(parameters, gradients, velocity, learning_rate=0.01, beta=0.9):
    velocity["dW1"] = beta * velocity["dW1"] - learning_rate * gradients["dW1"]
    velocity["db1"] = beta * velocity["db1"] - learning_rate * gradients["db1"]
    velocity["dW2"] = beta * velocity["dW2"] - learning_rate * gradients["dW2"]
    velocity["db2"] = beta * velocity["db2"] - learning_rate * gradients["db2"]

    parameters["W1"] = parameters["W1"] + velocity["dW1"]
    parameters["b1"] = parameters["b1"] + velocity["db1"]
    parameters["W2"] = parameters["W2"] + velocity["dW2"]
    parameters["b2"] = parameters["b2"] + velocity["db2"]

    return parameters, velocity

def train_neural_network_with_early_stopping(
    X_train,
    y_train,
    X_val,
    y_val,
    input_size,
    hidden_size,
    output_size,
    hidden_activation="relu",
    output_activation="softmax",
    loss_type="cross_entropy",
    class_weights=None,
    learning_rate=0.01,
    epochs=1000,
    print_every=100,
    patience=200,
    seed=42,
):
    if output_activation != "softmax":
        raise ValueError("Training requires a softmax output for the implemented gradients.")
    parameters = initialize_parameters(input_size, hidden_size, output_size, seed)
    velocity = initialize_velocity(parameters)
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_parameters = None
    patience_counter = 0

    for epoch in range(epochs):
        Y_hat_train, cache = forward_propagation(
            X_train,
            parameters,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

        train_loss = compute_loss(
            y_train,
            Y_hat_train,
            loss_type=loss_type,
            class_weights=class_weights
        )

        train_loss_history.append(train_loss)
        gradients = backward_propagation(
            y_train,
            parameters,
            cache,
            hidden_activation=hidden_activation,
            loss_type=loss_type,
            class_weights=class_weights
        )

        parameters, velocity = update_parameters_with_momentum(
            parameters,
            gradients,
            velocity,
            learning_rate=learning_rate,
            beta=0.9
        )

        Y_hat_val, _ = forward_propagation(
            X_val,
            parameters,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

        val_loss = compute_loss(
            y_val,
            Y_hat_val,
            loss_type=loss_type,
            class_weights=class_weights
        )
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_parameters = {
                "W1": parameters["W1"].copy(),
                "b1": parameters["b1"].copy(),
                "W2": parameters["W2"].copy(),
                "b2": parameters["b2"].copy()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if print_every and epoch % print_every == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            break

    return best_parameters, train_loss_history, val_loss_history

def predict(X, parameters, hidden_activation="relu", output_activation="softmax"):
    Y_hat, _ = forward_propagation(
        X,
        parameters,
        hidden_activation=hidden_activation,
        output_activation=output_activation
    )

    predictions = np.argmax(Y_hat, axis=1)
    return predictions
