"""Tune configurations on fixed development partitions, leaving final test sets sealed."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .data import CLASSES, DATASET, labels, load_data, prepare_development, split_data
from .evaluate import metrics
from .knn import predict_knn
from .verify import verify_python

FEATURES = ("calendar", "electrical-only", "electrical")
K_VALUES = tuple(range(1, 52, 2))
SEEDS = (42, 7, 101)
SCREEN_EPOCHS = 240
FINALIST_EPOCHS = 600
PATIENCE = 30
GRIDS = {
    "logistic": [
        {"learning_rate": rate, "l2": weight}
        for rate, weight in product((0.1, 0.3), (0.0, 0.001, 0.01))
    ],
    "neural": [
        {"hidden_size": width, "learning_rate": rate, "hidden_activation": activation}
        for width, rate, activation in product((16, 32), (0.05, 0.1), ("relu", "tanh"))
    ],
}


def calendar_keys(frame):
    dates = frame["date"].dt
    return list(zip(dates.dayofweek, dates.hour * 3600 + dates.minute * 60 + dates.second))


def schedule_predictions(train, validation, y_train):
    """Use the training majority by weekday and time, with two fallback levels."""
    keys = calendar_keys(train)
    grouped = {}
    slots = {}
    overall = np.bincount(y_train, minlength=len(CLASSES))
    for key, label in zip(keys, y_train):
        grouped.setdefault(key, np.zeros(len(CLASSES), dtype=int))[label] += 1
        slots.setdefault(key[1], np.zeros(len(CLASSES), dtype=int))[label] += 1
    predictions = [
        grouped.get(key, slots.get(key[1], overall)).argmax() for key in calendar_keys(validation)
    ]
    return np.asarray(predictions, dtype=int), grouped


def compact_fit(model, settings, arrays, epochs, seed):
    trial = verify_python(model, *arrays, epochs, PATIENCE, settings=settings, seed=seed)
    trace = trial["convergence"]
    for name in ("train_loss", "validation_loss"):
        loss = trace.pop(name)
        trace[name + "_first"] = loss[0]
        trace[name + "_last"] = loss[-1]
        trace[name + "_last_25_epoch_drop"] = loss[max(0, len(loss) - 26)] - loss[-1]
    return trial


def tune_python(model, arrays):
    def loss(trial):
        return trial["convergence"]["checkpoint_validation_loss"]

    grid = GRIDS[model]
    screening = [compact_fit(model, config, arrays, SCREEN_EPOCHS, SEEDS[0]) for config in grid]
    leaders = sorted(range(len(grid)), key=lambda index: loss(screening[index]))[:2]
    finalists = []
    for index in leaders:
        trials = [compact_fit(model, grid[index], arrays, FINALIST_EPOCHS, seed) for seed in SEEDS]
        finalists.append({
            "grid_index": index, "initialization_trials": trials,
            "mean_validation_loss": float(np.mean([loss(trial) for trial in trials])),
        })
    winner = min(finalists, key=lambda item: item["mean_validation_loss"])
    return {"screening": screening, "finalists": finalists, "selected": winner["initialization_trials"][0]}


def knn_grid(executable, x_train, y_train, x_val, y_val):
    orderings = [
        ("native", np.arange(len(x_train))),
        *((f"shuffle_{seed}", np.random.default_rng(seed).permutation(len(x_train))) for seed in SEEDS[1:]),
    ]
    scores = []
    times = {}
    for name, order in orderings:
        start = perf_counter()
        prediction = predict_knn(executable, x_train[order], y_train[order], x_val, K_VALUES)
        times[name] = perf_counter() - start
        scores.append([metrics(y_val, prediction[:, i]) for i in range(len(K_VALUES))])
    trials = []
    for index, k in enumerate(K_VALUES):
        by_order = {name: score[index] for (name, _), score in zip(orderings, scores)}
        trials.append({
            "k": k, "by_reference_order": by_order,
            "mean_macro_f1": float(np.mean([item["macro_f1"] for item in by_order.values()])),
        })
    selected = max(trials, key=lambda trial: trial["mean_macro_f1"])
    return {
        "training_seconds": 0.0, "query_seconds_by_order": times,
        "reference_rows": len(x_train), "selected_k": selected["k"], "trials": trials,
    }


def calendar_diagnostics(executable, train, validation, x_train, y_train, x_val, y_val, groups):
    ks = (3, 11, 19, 51)
    train_keys, validation_keys = calendar_keys(train), calendar_keys(validation)
    by_key = {}
    for index, key in enumerate(train_keys):
        by_key.setdefault(key, []).append(index)
    exact = np.asarray([len(by_key.get(key, [])) for key in validation_keys])
    native = predict_knn(executable, x_train, y_train, x_val, ks)
    reverse = predict_knn(executable, x_train[::-1], y_train[::-1], x_val, ks)
    # Undo only the MinMax transformation on the sine/cosine columns.
    processor, _, _ = prepare_development(train, validation, "calendar")
    scaler = processor.named_transformers_["numeric"]
    unscaled = []
    for block in (x_train, x_val):
        unscaled.append(np.column_stack((scaler.inverse_transform(block[:, :2]), block[:, 2:])))
    raw = predict_knn(executable, unscaled[0], y_train, unscaled[1], ks)
    neighbor_checks = []
    for i, k in enumerate(ks):
        enough = exact >= k
        queries = np.flatnonzero(enough)
        votes = [
            np.bincount(y_train[by_key[validation_keys[j]][:k]], minlength=len(CLASSES)).argmax()
            for j in queries
        ]
        np.testing.assert_array_equal(native[queries, i], votes)
        neighbor_checks.append({
            "k": k, "queries_with_at_least_k_exact_matches": int(enough.sum()),
            "native": metrics(y_val, native[:, i]), "reverse": metrics(y_val, reverse[:, i]),
            "predictions_changed_without_scaling": int(np.count_nonzero(raw[:, i] != native[:, i])),
        })

    def profiles(frame):
        return frame.groupby(frame.date.dt.normalize())["Load_Type"].agg(tuple).value_counts()

    train_profiles, val_profiles = profiles(train), profiles(validation)
    return {
        "train_class_counts": np.bincount(y_train, minlength=len(CLASSES)).tolist(),
        "validation_class_counts": np.bincount(y_val, minlength=len(CLASSES)).tolist(),
        "ambiguous_training_keys": int(sum(np.count_nonzero(counts) > 1 for counts in groups.values())),
        "training_keys": len(groups), "exact_neighbor_count_range": [int(exact.min()), int(exact.max())],
        "daily_profile_counts": [
            {"training_days": int(count), "validation_days": int(val_profiles.get(profile, 0))}
            for profile, count in train_profiles.items()
        ],
        "validation_days_with_unseen_profile": int(val_profiles[~val_profiles.index.isin(train_profiles.index)].sum()),
        "neighbor_checks": neighbor_checks,
    }


def tune(frame, executable):
    results = []
    for method in ("random", "chronological"):
        train, validation, _ = split_data(frame, method, SEEDS[0])
        y_train, y_val = labels(train), labels(validation)
        start = perf_counter()
        schedule, groups = schedule_predictions(train, validation, y_train)
        baseline = {
            "configuration": "weekday and time majority with time-only and overall fallbacks",
            "fit_and_predict_seconds": perf_counter() - start, "validation": metrics(y_val, schedule),
        }
        result = {"split": method, "schedule": baseline, "feature_runs": {}}
        for feature_set in FEATURES:
            processor, x_train, x_val = prepare_development(train, validation, feature_set)
            arrays = (x_train, y_train, x_val, y_val)
            if method == "chronological" and feature_set == "calendar":
                result["calendar_diagnostics"] = calendar_diagnostics(executable, train, validation, *arrays, groups)
            runs = {"columns": processor.get_feature_names_out().tolist()}
            runs["knn"] = knn_grid(executable, *arrays)
            for model in GRIDS:
                runs[model] = tune_python(model, arrays)
                print(f"Finished {method} / {feature_set} / {model}", flush=True)
            result["feature_runs"][feature_set] = runs
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--knn-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    with threadpool_limits(limits=1):
        results = tune(load_data(args.data), args.knn_executable)
    report = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "split_seed": SEEDS[0], "blas_threads": 1, "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "wall_seconds": perf_counter() - start, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()

