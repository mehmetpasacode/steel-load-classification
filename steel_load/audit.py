"""Inspect possible target proxies using only the chronological training period."""

import argparse
from itertools import groupby
from pathlib import Path

from .data import CLASSES, CO2, DATASET, TARGET, USAGE, VALIDATION_START, load_data


def audit(frame):
    train = frame[frame["date"] < VALIDATION_START].sort_values("date")
    daily_profiles = train.groupby(train["date"].dt.normalize())[TARGET].agg(tuple).value_counts()
    grouped = train.groupby(TARGET)[USAGE]
    purities = {}
    for column in (USAGE, CO2):
        counts = train.groupby([column, TARGET]).size().unstack(fill_value=0)
        purities[column] = counts.max(axis=1).sum() / len(train)
    return {
        "train_rows": len(train),
        "usage_by_class": grouped.agg(["count", "min", "median", "max"]).reindex(CLASSES),
        "usage_co2_pearson": train[USAGE].corr(train[CO2]),
        "in_sample_exact_value_purity": purities,
        "majority_class_fraction": train[TARGET].value_counts(normalize=True).max(),
        "daily_profiles": [
            {"days": int(count), "runs": [(label, sum(1 for _ in run)) for label, run in groupby(profile)]}
            for profile, count in daily_profiles.items()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET)
    args = parser.parse_args()
    for name, value in audit(load_data(args.data)).items():
        print(f"{name}\n{value}\n")


if __name__ == "__main__":
    main()

