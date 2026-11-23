"""NumPy implementation adapted from the original course project."""

import numpy as np

from .common import cross_entropy, softmax


def initialize_logistic_parameters(input_size, output_size, seed=42):

    rng = np.random.default_rng(seed)

    W = 0.01 * rng.standard_normal((input_size, output_size))
    b = np.zeros((1, output_size))

    parameters = {
        "W": W,
        "b": b
    }

    return parameters

def initialize_logistic_velocity(parameters):
    velocity = {
        "dW": np.zeros_like(parameters["W"]),
        "db": np.zeros_like(parameters["b"])
    }

    return velocity


def forward_logistic_regression(X, parameters):

    W = parameters["W"]
    b = parameters["b"]

    Z = np.dot(X, W) + b
    Y_hat = softmax(Z)

    cache = {
        "X": X,
        "Z": Z,
        "Y_hat": Y_hat
    }

    return Y_hat, cache


def backward_logistic_regression(y_true, cache):

    X = cache["X"]
    Y_hat = cache["Y_hat"]

    m = y_true.shape[0]

    dZ = Y_hat - y_true

    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m

    gradients = {
        "dW": dW,
        "db": db
    }

    return gradients


def update_logistic_parameters_with_momentum(
    parameters,
    gradients,
    velocity,
    learning_rate=0.01,
    beta=0.9
):
    velocity["dW"] = beta * velocity["dW"] - learning_rate * gradients["dW"]
    velocity["db"] = beta * velocity["db"] - learning_rate * gradients["db"]

    parameters["W"] = parameters["W"] + velocity["dW"]
    parameters["b"] = parameters["b"] + velocity["db"]

    return parameters, velocity


def train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    input_size,
    output_size,
    learning_rate=0.01,
    epochs=1000,
    print_every=100,
    patience=200,
    seed=42,
    l2=0.0,
):
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative.")
    parameters = initialize_logistic_parameters(
        input_size,
        output_size,
        seed,
    )
    velocity = initialize_logistic_velocity(parameters)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_parameters = None
    patience_counter = 0

    for epoch in range(epochs):

        Y_hat_train, cache = forward_logistic_regression(
            X_train,
            parameters
        )

        train_loss = cross_entropy(
            y_train,
            Y_hat_train,
        )
        train_loss += 0.5 * l2 * np.sum(parameters["W"] ** 2)

        train_loss_history.append(train_loss)

        gradients = backward_logistic_regression(
            y_train,
            cache
        )
        gradients["dW"] += l2 * parameters["W"]

        parameters, velocity = update_logistic_parameters_with_momentum(
            parameters,
            gradients,
            velocity,
            learning_rate=learning_rate,
            beta=0.9
        )

        Y_hat_val, _ = forward_logistic_regression(
            X_val,
            parameters
        )

        val_loss = cross_entropy(
            y_val,
            Y_hat_val,
        )

        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_parameters = {
                "W": parameters["W"].copy(),
                "b": parameters["b"].copy()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if print_every and epoch % print_every == 0:
            print(
                f"Epoch {epoch}, "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )

        if patience_counter >= patience:
            break

    return best_parameters, train_loss_history, val_loss_history


def predict_logistic_regression(X, parameters):

    Y_hat, _ = forward_logistic_regression(
        X,
        parameters
    )

    predictions = np.argmax(Y_hat, axis=1)

    return predictions
