"""Select a model on validation, then score the held-out test split once."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from . import logistic_regression as logistic
from . import neural_network as neural
from .data import CLASSES, DATASET, FEATURE_SETS, feature_frame, labels, load_data
from .data import prepare_development, split_data
from .knn import predict_knn


def metrics(truth, predictions):
    order = list(range(len(CLASSES)))
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, labels=order, average="macro", zero_division=0),
        "per_class_recall": dict(zip(
            CLASSES, recall_score(truth, predictions, labels=order, average=None, zero_division=0)
        )),
        "class_order": CLASSES,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=order).tolist(),
    }


def select_python(model, x_train, y_train, x_val, y_val, epochs, patience, seed):
    targets = np.eye(len(CLASSES))
    shared = {
        "input_size": x_train.shape[1],
        "output_size": len(CLASSES),
        "epochs": epochs,
        "patience": patience,
        "print_every": 0,
        "seed": seed,
    }
    if model == "logistic":
        candidates = [{"learning_rate": rate} for rate in (0.01, 0.05, 0.1)]
        train = logistic.train_logistic_regression
        predict = logistic.predict_logistic_regression
    else:
        candidates = [
            {"hidden_size": width, "hidden_activation": activation, "learning_rate": 0.1}
            for width in (16, 32, 64)
            for activation in ("relu", "tanh", "leaky_relu")
        ]
        train = neural.train_neural_network_with_early_stopping

    trials = []
    best_accuracy = -1.0
    best_model = None
    for settings in candidates:
        params, _, losses = train(
            x_train, targets[y_train], x_val, targets[y_val], **shared, **settings
        )
        if params is None:
            raise ValueError("Training produced no finite validation checkpoint.")
        if model == "neural":
            predictions = neural.predict(x_val, params, settings["hidden_activation"])
        else:
            predictions = predict(x_val, params)
        score = accuracy_score(y_val, predictions)
        trials.append({
            **settings,
            "checkpoint_epoch": int(np.argmin(losses)) + 1,
            "validation_accuracy": score,
        })
        if score > best_accuracy:
            best_accuracy = score
            best_model = (settings, params)

    settings, params = best_model
    if model == "neural":
        def predict_selected(features):
            return neural.predict(features, params, settings["hidden_activation"])
    else:
        def predict_selected(features):
            return logistic.predict_logistic_regression(features, params)
    return settings, trials, predict_selected


def select_knn(executable, x_train, y_train, x_val, y_val, neighbors):
    candidates = sorted(set(neighbors))
    validation = predict_knn(executable, x_train, y_train, x_val, candidates)
    trials = [
        {"k": k, "validation_accuracy": accuracy_score(y_val, validation[:, index])}
        for index, k in enumerate(candidates)
    ]
    best = max(trials, key=lambda trial: trial["validation_accuracy"])
    k = best["k"]

    def predict_selected(features):
        return predict_knn(executable, x_train, y_train, features, [k])[:, 0]

    return {"k": k}, trials, predict_selected


def evaluate(args):
    train, validation, test = split_data(load_data(args.data), args.split, args.seed)
    preprocessor, x_train, x_val = prepare_development(train, validation, args.features)
    y_train, y_val = labels(train), labels(validation)
    if args.model == "knn":
        settings, trials, predict = select_knn(
            args.knn_executable, x_train, y_train, x_val, y_val, args.k
        )
    else:
        settings, trials, predict = select_python(
            args.model, x_train, y_train, x_val, y_val, args.epochs, args.patience, args.seed
        )
    # Test data enters preprocessing and prediction only after selection.
    x_test = preprocessor.transform(feature_frame(test, args.features))
    return {
        "model": args.model,
        "split": args.split,
        "feature_set": args.features,
        "seed": args.seed,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected": settings,
        "validation_trials": trials,
        "test": metrics(labels(test), predict(x_test)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("logistic", "neural", "knn"), required=True)
    parser.add_argument("--split", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--features", choices=FEATURE_SETS, default="calendar")
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--k", nargs="+", type=int, default=[3, 11, 19, 35])
    parser.add_argument("--knn-executable", type=Path)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result")
    args = parser.parse_args()
    if args.model == "knn" and args.knn_executable is None:
        parser.error("--knn-executable is required for k-NN")
    if args.epochs < 1 or args.patience < 1 or not 0 <= args.seed < 2**32:
        parser.error("epochs and patience must be positive and seed must fit in an unsigned 32-bit integer")
    try:
        result = json.dumps(evaluate(args), indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n", encoding="utf8")
        print(result)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()

