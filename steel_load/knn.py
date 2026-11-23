"""Pass train labels and row features to the C++17 neighbor engine."""

import subprocess
from io import StringIO
from pathlib import Path

import numpy as np

from .data import CLASSES


def predict_knn(executable, x_train, y_train, queries, neighbors):
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise ValueError(f"C++ executable not found: {executable}. See the README for build commands.")
    if len(x_train) != len(y_train):
        raise ValueError("Training columns and labels have different row counts.")
    if not neighbors or min(neighbors) < 1 or max(neighbors) > len(x_train):
        raise ValueError("Each k must be between 1 and the number of training rows.")

    blocks = StringIO()
    blocks.write(f"{len(x_train)} {len(queries)} {x_train.shape[1]} {len(CLASSES)} {len(neighbors)}\n")
    blocks.write(" ".join(map(str, neighbors)) + "\n")
    np.savetxt(blocks, y_train.reshape(1, -1), fmt="%d")
    np.savetxt(blocks, x_train, fmt="%.17g")
    np.savetxt(blocks, queries, fmt="%.17g")
    result = subprocess.run(
        [str(executable)], input=blocks.getvalue(), capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError(f"C++ k-NN failed: {result.stderr.strip()}")
    predictions = np.loadtxt(StringIO(result.stdout), dtype=int, ndmin=2)
    if predictions.shape != (len(queries), len(neighbors)):
        raise ValueError("Unexpected prediction shape from C++ k-NN.")
    if np.any((predictions < 0) | (predictions >= len(CLASSES))):
        raise ValueError("Unexpected class ID from C++ k-NN.")
    return predictions

