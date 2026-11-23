import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from steel_load.data import (
    CLASSES, CO2, DATASET, TARGET, TEST_START, USAGE, VALIDATION_START, feature_frame,
    labels, load_data, make_preprocessor, split_data,
)
from steel_load.evaluate import evaluate, metrics, select_knn
from steel_load.knn import predict_knn
from steel_load.logistic_regression import initialize_logistic_parameters, train_logistic_regression
from steel_load.tune import schedule_predictions
from steel_load.verify import majority_class, verify_python


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data(DATASET)

    def test_chronological_boundaries_and_coverage(self):
        parts = split_data(self.frame, "chronological")
        train, validation, test = parts
        self.assertEqual([len(part) for part in parts], [24480, 3552, 7008])
        self.assertTrue(all(part.date.is_monotonic_increasing for part in parts))
        self.assertLess(train.date.max(), VALIDATION_START)
        self.assertEqual(validation.date.min(), VALIDATION_START)
        self.assertLess(validation.date.max(), TEST_START)
        self.assertEqual(test.date.min(), TEST_START)
        self.assertEqual(len(set().union(*(part.index for part in parts))), 35040)

    def test_random_split_is_stratified_and_reproducible(self):
        parts = split_data(self.frame, "random")
        self.assertEqual([len(part) for part in parts], [24528, 3504, 7008])
        self.assertEqual(len(set().union(*(part.index for part in parts))), 35040)
        fractions = self.frame[TARGET].value_counts(normalize=True)
        for first, second in zip(parts, split_data(self.frame, "random")):
            self.assertTrue(first.index.equals(second.index))
            proportions = first[TARGET].value_counts(normalize=True).reindex(fractions.index)
            np.testing.assert_allclose(proportions, fractions, atol=0.001)

    def test_preprocessor_learns_only_training_ranges_and_categories(self):
        sample = self.frame.iloc[:3].copy()
        sample["date"] = pd.to_datetime(
            ["2018-01-01 00:00", "2018-01-01 12:15", "2018-01-06 00:00"],
            format="%Y-%m-%d %H:%M",
        )
        sample[USAGE] = [10, 20, 30]
        proc = make_preprocessor("interval")
        proc.fit(feature_frame(sample.iloc[:2], "interval"))
        output = proc.transform(feature_frame(sample.iloc[2:], "interval"))
        self.assertEqual(output[0, 0], 2.0)
        np.testing.assert_array_equal(output[0, -2:], [0, 0])
        self.assertNotIn(TARGET, feature_frame(sample, "interval"))
        self.assertTrue({USAGE, CO2}.isdisjoint(feature_frame(sample, "electrical")))
        self.assertEqual(set(feature_frame(sample[["date"]], "calendar")),
                         {"WeekStatus", "Day_of_week", "NSM_sin", "NSM_cos"})

    def test_matrix_uses_the_same_ordinal_encoding_on_both_axes(self):
        truth = labels(pd.DataFrame({TARGET: ["Maximum_Load", "Light_Load", "Medium_Load"]}))
        np.testing.assert_array_equal(truth, [2, 0, 1])
        result = metrics(truth, [1, 0, 2])
        self.assertEqual(result["class_order"], CLASSES)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.assertEqual(metrics([0], [0])["confusion_matrix"], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(list(result["per_class_recall"].values()), [1.0, 0.0, 0.0])

    def test_majority_uses_training_labels_and_fixed_tie_order(self):
        self.assertEqual(majority_class(np.array([2, 2, 1])), 2)
        self.assertEqual(majority_class(np.array([2, 1])), 1)

    def test_schedule_baseline_uses_specific_slot_then_falls_back(self):
        train = pd.DataFrame({"date": pd.to_datetime([
            "2018-01-01 09:00", "2018-01-08 09:00", "2018-01-02 09:00", "2018-01-02 10:00",
        ])})
        validation = pd.DataFrame({"date": pd.to_datetime([
            "2018-01-09 09:00", "2018-01-10 09:00", "2018-01-10 11:00",
        ])})
        predictions, _ = schedule_predictions(train, validation, np.array([1, 1, 2, 0]))
        np.testing.assert_array_equal(predictions, [2, 1, 1])

    def test_electrical_only_uses_four_meter_columns(self):
        sample = self.frame.iloc[:8].drop(columns="date")
        features = feature_frame(sample, "electrical-only")
        fitted = make_preprocessor("electrical-only").fit_transform(features)
        self.assertEqual(fitted.shape, (8, 4))
        self.assertEqual(len(features.columns), 4)
        self.assertNotIn(TARGET, features.columns)

    def test_logistic_l2_penalizes_only_weights(self):
        features = np.ones((3, 2))
        targets = np.eye(3)
        options = dict(input_size=2, output_size=3, epochs=1, learning_rate=0.1, print_every=0)
        initial = initialize_logistic_parameters(2, 3)
        plain, loss_plain, _ = train_logistic_regression(features, targets, features, targets, **options)
        regularized, loss_l2, _ = train_logistic_regression(
            features, targets, features, targets, l2=0.01, **options
        )
        np.testing.assert_allclose(regularized["W"], plain["W"] - 0.001 * initial["W"])
        np.testing.assert_array_equal(regularized["b"], plain["b"])
        self.assertAlmostEqual(loss_l2[0] - loss_plain[0], 0.005 * np.sum(initial["W"] ** 2))

    def test_early_stopping_restores_checkpoint_when_validation_loss_rises(self):
        features = np.ones((6, 2))
        y_train = np.zeros(6, dtype=int)
        y_val = np.ones(6, dtype=int)
        for model in ("logistic", "neural"):
            with self.subTest(model=model):
                result = verify_python(model, features, y_train, features, y_val, 12, 3)
                trace = result["convergence"]
                self.assertEqual(trace["stop_reason"], "patience")
                self.assertEqual(trace["epochs_run"], 4)
                self.assertEqual(trace["checkpoint_epoch"], 1)
                self.assertLess(trace["train_loss"][-1], trace["train_loss"][0])
                self.assertGreater(trace["validation_loss"][-1], trace["checkpoint_validation_loss"])

    def test_python_models_on_tiny_fixture(self):
        self.run_tiny_evaluations(("logistic", "neural"))

    def run_tiny_evaluations(self, models, executable=None):
        frames = []
        for day in ["12/09/2018", "13/09/2018", "20/10/2018"]:
            frames.append(pd.DataFrame({
                "date": [f"{day} {hour:02d}:15" for hour in range(9)],
                TARGET: CLASSES * 3,
            }))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            pd.concat(frames).to_csv(path, index=False)
            for model in models:
                with self.subTest(model=model):
                    result = evaluate(Namespace(
                        model=model, split="chronological", features="calendar", data=path,
                        seed=42, epochs=2, patience=2, k=[1, 3], knn_executable=executable,
                    ))
                    self.assertEqual(result["split_sizes"], {"train": 9, "validation": 9, "test": 9})
                    trials = result["validation_trials"]
                    best = max(trials, key=lambda trial: trial["validation_accuracy"])
                    self.assertEqual(result["selected"], {key: best[key] for key in result["selected"]})
                    self.assertEqual(np.sum(result["test"]["confusion_matrix"]), 9)

    def test_knn_only_queries_test_rows_with_selected_k(self):
        train = np.array([[0.0], [1.0], [2.0]])
        validation = np.array([[3.0], [4.0]])
        test = np.array([[5.0]])
        with patch("steel_load.evaluate.predict_knn", side_effect=[
            np.array([[2, 1], [2, 1]]), np.array([[0]])
        ]) as backend:
            settings, _, predict = select_knn("unused", train, np.arange(3), validation, [1, 1], [3, 1])
            self.assertEqual(settings, {"k": 3})
            self.assertEqual(backend.call_count, 1)
            self.assertIs(backend.call_args.args[3], validation)
            np.testing.assert_array_equal(predict(test), [0])
            self.assertIs(backend.call_args.args[3], test)
            self.assertEqual(backend.call_args.args[4], [3])
            self.assertTrue(all(call.args[1] is train for call in backend.call_args_list))

    @unittest.skipUnless(os.environ.get("STEEL_KNN_EXECUTABLE"), "C++ executable not configured")
    def test_cpp_exact_matches_ties_and_end_to_end_pipeline(self):
        executable = os.environ["STEEL_KNN_EXECUTABLE"]
        prediction = predict_knn(
            executable, np.array([[0.0], [2.0], [2.0]]), np.array([2, 1, 0]),
            np.array([[0.0], [2.0], [1.0]]), [2, 1, 3],
        )
        np.testing.assert_array_equal(prediction, [[1, 2, 0], [0, 1, 0], [1, 2, 0]])
        self.run_tiny_evaluations(("knn",), executable)

