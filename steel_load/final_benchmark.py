"""Score the frozen benchmark, with all checkpoints checked before holdout scoring."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from . import logistic_regression as logistic
from . import neural_network as neural
from .data import CLASSES, DATASET, feature_frame, labels, load_data, prepare_development, split_data
from .encoding_experiment import encoding_columns
from .evaluate import metrics
from .knn import predict_knn
from .tune import schedule_predictions
from .verify import majority_class

DOCS = DATASET.parents[1] / "docs"
SEED = 42
FEATURES = "electrical"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(data_path):
    neural_path = DOCS / "encoding-extended-frozen.json"
    tuning_path = DOCS / "selected-configurations.json"
    network = json.loads(neural_path.read_text(encoding="utf8"))
    tuning = json.loads(tuning_path.read_text(encoding="utf8"))
    source_path = DOCS / "encoding-extended.json"
    dataset_hash = sha256(data_path)
    if (network["source"] != source_path.name or network["source_sha256"] != sha256(source_path)
            or network["encoding"] != "C" or dataset_hash != network["dataset_sha256"]
            or dataset_hash != tuning["dataset_sha256"]
            or network["split_seed"] != SEED or tuning["split_seed"] != SEED):
        raise ValueError("Frozen benchmark provenance did not match.")
    source = json.loads(source_path.read_text(encoding="utf8"))
    plan = {}
    for method in ("random", "chronological"):
        nn = next(row for row in network["fixed_runs"] if row["split"] == method)
        row = next(row for row in tuning["splits"] if row["split"] == method)
        trial = next(group for group in source["results"] if group["split"] == method)
        record = next(item for item in trial["arms"]["C"] if item["configuration"]["seed"] == SEED)
        if (nn["features"] != FEATURES or nn["model"] != "neural"
                or nn["configuration"]["epoch_cap"] != 2000 or nn["configuration"]["seed"] != SEED
                or nn["class_order"] != list(CLASSES)
                or nn["encoded_columns"] != trial["encoded_columns"]["C"]
                or nn["configuration"] != record["configuration"]
                or nn["checkpoint_epoch"] != record["convergence"]["checkpoint_epoch"]):
            raise ValueError(f"Invalid neural freeze for {method}.")
        plan[method] = {
            "neural": nn,
            **{name: next(item for item in row["selected_by_model_and_features"]
                          if item["model"] == name and item["features"] == FEATURES)
               for name in ("logistic", "knn")},
            "schedule": row["advance"],
        }
        knn = plan[method]["knn"]["configuration"]
        if knn["reference_order"] != "native" or knn["distance"] != "squared Euclidean":
            raise ValueError(f"Unexpected k-NN freeze for {method}.")
    return plan, {
        "dataset_sha256": dataset_hash,
        "frozen_source_sha256": {path.name: sha256(path) for path in (neural_path, tuning_path, source_path)},
    }


def fit_python(model, entry, x_train, y_train, x_val, y_val):
    settings = entry["configuration"].copy()
    if (settings.pop("seed") != SEED or settings.pop("momentum") != 0.9
            or settings.pop("loss") != "cross_entropy"):
        raise ValueError(f"Unexpected {model} optimizer settings.")
    epochs, patience = settings.pop("epoch_cap"), settings.pop("patience")
    target = np.eye(len(CLASSES))
    if model == "neural":
        train = neural.train_neural_network_with_early_stopping
        predict = partial(neural.predict, hidden_activation=settings["hidden_activation"])
    else:
        train = logistic.train_logistic_regression
        predict = logistic.predict_logistic_regression
    start = perf_counter()
    params, _, loss = train(
        x_train, target[y_train], x_val, target[y_val],
        input_size=x_train.shape[1], output_size=len(CLASSES),
        epochs=epochs, patience=patience, seed=SEED, print_every=0, **settings,
    )
    seconds = perf_counter() - start
    if params is None or not np.isfinite(loss).all():
        raise ValueError(f"Invalid {model} training result.")
    checkpoint = int(np.argmin(loss)) + 1
    if checkpoint != entry["checkpoint_epoch"]:
        raise ValueError(f"{model} did not reproduce its frozen checkpoint epoch.")
    infer = partial(predict, parameters=params)
    score = metrics(y_val, infer(x_val))
    expected = ({"accuracy": entry["validation_accuracy"], "macro_f1": entry["validation_macro_f1"]}
                if model == "neural" else entry["reference_validation"])
    np.testing.assert_allclose(
        [score["accuracy"], score["macro_f1"]], [expected["accuracy"], expected["macro_f1"]],
        atol=1e-12, rtol=0,
    )
    return infer, {
        "configuration": entry["configuration"], "fit_seconds": seconds,
        "checkpoint_epoch": checkpoint, "epochs_run": len(loss),
        "checkpoint_validation_loss": loss[checkpoint - 1],
        "reproduced_validation": score,
    }


def prepare_split(frame, method, plan, executable):
    train, validation, test = split_data(frame, method, SEED)
    if (not train.index.intersection(validation.index).empty
            or not train.index.intersection(test.index).empty
            or not validation.index.intersection(test.index).empty):
        raise ValueError("Split partitions overlap.")
    start = perf_counter()
    proc, x_train, x_val = prepare_development(train, validation, FEATURES)
    prep_seconds = perf_counter() - start
    # Check the fitted state directly against the training frame.
    train_features = feature_frame(train, FEATURES)
    numeric = proc.transformers_[0][2]
    scaler = proc.named_transformers_["numeric"]
    np.testing.assert_equal(scaler.n_samples_seen_, len(train))
    np.testing.assert_array_equal(scaler.data_min_, train_features[numeric].min().to_numpy())
    np.testing.assert_array_equal(scaler.data_max_, train_features[numeric].max().to_numpy())
    for column, fitted in zip(proc.transformers_[1][2], proc.named_transformers_["calendar_categories"].categories_):
        np.testing.assert_array_equal(fitted, np.unique(train_features[column]))
    names = proc.get_feature_names_out()
    c = encoding_columns(names, "C")
    if len(names) != 15 or len(c) != 12 or names[c].tolist() != plan["neural"]["encoded_columns"]:
        raise ValueError("Encoded feature columns do not match the freeze.")
    y_train, y_val = labels(train), labels(validation)
    neural_predict, neural_info = fit_python(
        "neural", plan["neural"], np.ascontiguousarray(x_train[:, c]), y_train,
        np.ascontiguousarray(x_val[:, c]), y_val,
    )
    logistic_predict, logistic_info = fit_python("logistic", plan["logistic"], x_train, y_train, x_val, y_val)
    k = plan["knn"]["configuration"]["k"]
    knn_score = metrics(y_val, predict_knn(executable, x_train, y_train, x_val, [k])[:, 0])
    schedule_score = metrics(y_val, schedule_predictions(
        train[["date"]], validation[["date"]], y_train
    )[0])
    for name, score in (("knn", knn_score), ("schedule", schedule_score)):
        if score["confusion_matrix"] != plan[name]["reference_validation"]["confusion_matrix"]:
            raise ValueError(f"{name} did not reproduce its frozen validation confusion matrix.")
    start = perf_counter()
    majority = majority_class(y_train)
    majority_fit_seconds = perf_counter() - start
    return {
        "split": method, "train": train, "validation_rows": len(validation), "test": test,
        "proc": proc, "prep_seconds": prep_seconds, "c": c, "names": names.tolist(),
        "x_train": x_train, "y_train": y_train, "majority": majority, "majority_fit_seconds": majority_fit_seconds,
        "neural_predict": neural_predict, "neural_info": neural_info,
        "logistic_predict": logistic_predict, "logistic_info": logistic_info, "plan": plan,
        "knn_validation": knn_score, "schedule_validation": schedule_score,
    }


def score_split(state, executable):
    test = state["test"]
    plan = state["plan"]
    start = perf_counter()
    x_test = state["proc"].transform(feature_frame(test, FEATURES))
    transform_seconds = perf_counter() - start
    predictions = {}
    records = {}

    def collect(name, predictor, record, time_key="prediction_seconds"):
        start = perf_counter()
        predictions[name] = predictor()
        records[name] = {**record, time_key: perf_counter() - start}

    collect("neural", lambda: state["neural_predict"](np.ascontiguousarray(x_test[:, state["c"]])),
            {**state["neural_info"], "features": FEATURES, "encoding": "C", "input_columns": 12})
    collect("logistic", lambda: state["logistic_predict"](x_test),
            {**state["logistic_info"], "features": FEATURES, "input_columns": 15})
    k = plan["knn"]["configuration"]["k"]
    collect("knn", lambda: predict_knn(executable, state["x_train"], state["y_train"], x_test, [k])[:, 0],
            {"configuration": plan["knn"]["configuration"], "features": FEATURES, "input_columns": 15,
             "fit_seconds": 0.0, "reference_rows": len(state["x_train"]),
             "reproduced_validation": state["knn_validation"]})
    collect("majority", lambda: np.full(len(test), state["majority"]),
            {"configuration": {"label": CLASSES[state["majority"]]}, "features": "none",
             "fit_seconds": state["majority_fit_seconds"]})
    collect("calendar_schedule", lambda: schedule_predictions(
        state["train"][["date"]], test[["date"]], state["y_train"]
    )[0], {"configuration": plan["schedule"]["configuration"], "features": "calendar",
           "reproduced_validation": state["schedule_validation"]},
            time_key="fit_and_prediction_seconds")

    truth = labels(test)
    for name, prediction in predictions.items():
        records[name]["test"] = metrics(truth, prediction)
    return {
        "split": state["split"],
        "rows": {"train": len(state["train"]), "validation": state["validation_rows"], "test": len(test)},
        "preprocessing": {"fit_and_validation_transform_seconds": state["prep_seconds"],
                          "test_transform_seconds": transform_seconds,
                          "fit_rows": len(state["train"]), "combined_columns": state["names"]},
        "models": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--knn-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DOCS / "final-benchmark.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. The final holdout is already reserved or scored.")
    if not args.knn_executable.is_file():
        parser.error("C++ k-NN executable does not exist.")
    start = perf_counter()
    plan, provenance = load_plan(args.data)
    frame = load_data(args.data)
    with threadpool_limits(limits=1):
        states = []
        for method in plan:
            states.append(prepare_split(frame, method, plan[method], args.knn_executable))
            print(f"Verified {method} frozen checkpoints and training-only preprocessing.", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as marker:
            json.dump({"status": "holdout_scoring_started", **provenance}, marker)
        results = [score_split(state, args.knn_executable) for state in states]
    report = {
        "status": "complete", **provenance, "seed": SEED, "blas_threads": 1,
        "knn_executable_sha256": sha256(args.knn_executable),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "wall_seconds": perf_counter() - start, "results": results,
    }
    pending = args.output.with_suffix(args.output.suffix + ".tmp")
    with pending.open("x", encoding="utf8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    pending.replace(args.output)
    print(f"Saved both held-out partitions to {args.output}", flush=True)


if __name__ == "__main__":
    main()

