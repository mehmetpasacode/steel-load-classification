"""Compare calendar encodings on fixed validation splits with frozen model settings."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .data import DATASET, labels, load_data, prepare_development, split_data
from .verify import verify_python

FROZEN = DATASET.parents[1] / "docs" / "selected-configurations.json"
SEEDS = (42, 7, 101)
ENCODINGS = {
    "A": "seven day columns and two WeekStatus columns",
    "B": "seven day columns",
    "C": "six day columns with Monday as reference",
}
FEATURES = ("calendar", "electrical")


def encoding_columns(names, arm):
    if arm not in ENCODINGS:
        raise ValueError(f"Unknown encoding: {arm}")
    return [
        i for i, name in enumerate(names)
        if (arm == "A" or not name.startswith("calendar_categories__WeekStatus_"))
        and (arm != "C" or name != "calendar_categories__Day_of_week_Monday")
    ]


def run_experiment(frame, frozen, *, extended=False):
    feature_sets = ("electrical",) if extended else FEATURES
    models = ("neural",) if extended else ("logistic", "neural")
    encodings = ("A", "C") if extended else tuple(ENCODINGS)
    results = []
    for split in frozen["splits"]:
        method = split["split"]
        train, validation, _ = split_data(frame, method, frozen["split_seed"])
        y_train, y_val = labels(train), labels(validation)
        for features in feature_sets:
            processor, x_train, x_val = prepare_development(train, validation, features)
            names = processor.get_feature_names_out()
            columns = {arm: encoding_columns(names, arm) for arm in encodings}
            matrices = {
                arm: (np.ascontiguousarray(x_train[:, indices]),
                      np.ascontiguousarray(x_val[:, indices]))
                for arm, indices in columns.items()
            }
            for model in models:
                choice = next(
                    item for item in split["selected_by_model_and_features"]
                    if item["features"] == features and item["model"] == model
                )
                settings = choice["configuration"].copy()
                epochs, patience = settings.pop("epoch_cap"), settings.pop("patience")
                if extended:
                    epochs = 2000
                settings.pop("seed")
                if settings.pop("momentum") != 0.9 or settings.pop("loss") != "cross_entropy":
                    raise ValueError("Frozen momentum or loss differs from the development trainer.")
                arms = {arm: [] for arm in encodings}
                for offset, seed in enumerate(SEEDS):
                    offset %= len(encodings)
                    order = encodings
                    # Rotate execution order to reduce systematic timing effects.
                    for arm in order[offset:] + order[:offset]:
                        fit_train, fit_val = matrices[arm]
                        fit = verify_python(
                            model, fit_train, y_train, fit_val, y_val, epochs, patience,
                            settings=settings, seed=seed,
                        )
                        arms[arm].append(fit)
                # One loss target per split, feature set and model, anchored to the control.
                threshold = median(
                    fit["convergence"]["checkpoint_validation_loss"] for fit in arms["A"]
                ) + 0.01
                for fits in arms.values():
                    for fit in fits:
                        curve = fit["convergence"]["validation_loss"]
                        fit["first_epoch_at_common_loss"] = next(
                            (epoch for epoch, loss in enumerate(curve, 1) if loss <= threshold),
                            None,
                        )
                results.append({
                    "split": method, "features": features, "model": model,
                    "training_rows": len(train), "validation_rows": len(validation),
                    "encoded_columns": {
                        arm: names[indices].tolist() for arm, indices in columns.items()
                    },
                    "common_validation_loss_target": threshold, "arms": arms,
                })
                print(f"Completed {method} / {features} / {model}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true",
                        help="Compare combined-feature neural A and C at 2,000 epochs.")
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf8"))
    fingerprint = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if fingerprint != frozen["dataset_sha256"]:
        parser.error("Dataset fingerprint differs from the frozen configurations.")
    if args.output.exists():
        parser.error("Output already exists. Choose a new path for this run.")

    start = perf_counter()
    with threadpool_limits(limits=1):
        results = run_experiment(load_data(args.data), frozen, extended=args.extended)
    report = {
        "dataset_sha256": fingerprint, "split_seed": frozen["split_seed"],
        "initialization_seeds": SEEDS,
        "encodings": {arm: ENCODINGS[arm] for arm in results[0]["arms"]},
        "blas_threads": 1,
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "wall_seconds": perf_counter() - start, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()

