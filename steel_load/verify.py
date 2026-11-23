"""Run a fixed, validation-only pilot on the two shared splits."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from . import logistic_regression as logistic
from . import neural_network as neural
from .common import cross_entropy
from .data import (
    CLASSES, DATASET, labels, load_data, prepare_development, split_data,
)
from .evaluate import metrics
from .knn import predict_knn

K_VALUES = (3, 11, 19)
SEED = 42


def majority_class(y_train):
    return int(np.bincount(y_train, minlength=len(CLASSES)).argmax())


def verify_python(model, x_train, y_train, x_val, y_val, epochs, patience,
                  settings=None, seed=SEED):
    targets = np.eye(len(CLASSES))
    config = {"learning_rate": 0.1}
    if model == "neural":
        config.update(hidden_size=16, hidden_activation="relu")
        train = neural.train_neural_network_with_early_stopping
    else:
        train = logistic.train_logistic_regression
    if settings is not None:
        config.update(settings)

    start = perf_counter()
    params, train_loss, val_loss = train(
        x_train, targets[y_train], x_val, targets[y_val],
        input_size=x_train.shape[1], output_size=len(CLASSES),
        epochs=epochs, patience=patience, print_every=0, seed=seed, **config,
    )
    seconds = perf_counter() - start

    if params is None or not np.isfinite([train_loss, val_loss]).all():
        raise RuntimeError(f"{model} produced a non-finite loss")
    best = int(np.argmin(val_loss))
    if model == "neural":
        probabilities, _ = neural.forward_propagation(x_val, params, config["hidden_activation"])
    else:
        probabilities, _ = logistic.forward_logistic_regression(x_val, params)
    checkpoint_loss = cross_entropy(targets[y_val], probabilities)
    np.testing.assert_allclose(checkpoint_loss, val_loss[best], rtol=1e-12, atol=1e-12)
    patience_triggered = len(val_loss) - 1 - best >= patience
    if len(val_loss) < epochs and not patience_triggered:
        raise RuntimeError(f"{model} stopped before either stopping criterion was met")
    return {
        "model": model,
        "configuration": {
            **config, "momentum": 0.9, "loss": "cross_entropy",
            "epoch_cap": epochs, "patience": patience, "seed": seed,
        },
        "training_seconds": seconds,
        "validation": metrics(y_val, probabilities.argmax(axis=1)),
        "convergence": {
            "epochs_run": len(val_loss),
            "checkpoint_epoch": best + 1,
            "stop_reason": "patience" if patience_triggered else "epoch_cap",
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "checkpoint_validation_loss": checkpoint_loss,
        },
    }


def verify(frame, executable, epochs=300, patience=20):
    results = []
    for method in ("random", "chronological"):
        train, validation, _ = split_data(frame, method, SEED)
        y_train, y_val = labels(train), labels(validation)
        start = perf_counter()
        majority = majority_class(y_train)
        fit_seconds = perf_counter() - start
        split_result = {
            "split": method, "training_rows": len(train), "validation_rows": len(validation),
            "majority": {
                "configuration": {"label": CLASSES[majority]},
                "training_seconds": fit_seconds,
                "validation": metrics(y_val, np.full(len(y_val), majority)),
            },
            "feature_runs": [],
        }

        for feature_set in ("calendar", "electrical", "interval"):
            start = perf_counter()
            processor, x_train, x_val = prepare_development(train, validation, feature_set)
            preprocess_seconds = perf_counter() - start
            runs = [
                verify_python(model, x_train, y_train, x_val, y_val, epochs, patience)
                for model in ("logistic", "neural")
            ]
            start = perf_counter()
            predictions = predict_knn(executable, x_train, y_train, x_val, K_VALUES)
            prediction_seconds = perf_counter() - start
            knn = {
                "model": "knn",
                "configuration": {
                    "k_candidates": K_VALUES, "distance": "squared Euclidean",
                    "reference_rows": len(x_train), "query_rows": len(x_val),
                },
                "training_seconds": 0.0,
                "validation_seconds": prediction_seconds,
                "validation_trials": [
                    {"k": k, **metrics(y_val, predictions[:, index])}
                    for index, k in enumerate(K_VALUES)
                ],
            }
            knn["selected_k"] = max(knn["validation_trials"], key=lambda trial: trial["accuracy"])["k"]
            split_result["feature_runs"].append({
                "feature_set": feature_set,
                "feature_columns": processor.get_feature_names_out().tolist(),
                "preprocessing_seconds": preprocess_seconds,
                "models": [*runs, knn],
            })
            print(f"Completed {method} / {feature_set}", flush=True)
        results.append(split_result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--knn-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    start = perf_counter()
    with threadpool_limits(limits=1):
        results = verify(load_data(args.data), args.knn_executable, args.epochs, args.patience)
    artifact = {
        "seed": SEED, "blas_threads": 1,
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "wall_seconds": perf_counter() - start,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()

