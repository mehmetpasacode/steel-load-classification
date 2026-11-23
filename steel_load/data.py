"""Dataset schema, shared split policy, and train-fitted preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

DATASET = Path(__file__).resolve().parents[1] / "data" / "Steel_industry_data.csv"
CLASSES = ("Light_Load", "Medium_Load", "Maximum_Load")
TARGET = "Load_Type"
USAGE = "Usage_kWh"
CO2 = "CO2(tCO2)"
ELECTRICAL = (
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
)
CATEGORICAL = ("WeekStatus", "Day_of_week")
FEATURE_SETS = {
    "calendar": (),
    "electrical": ELECTRICAL,
    "interval": (USAGE, CO2, *ELECTRICAL),
    "electrical-only": ELECTRICAL,
}
VALIDATION_START = pd.Timestamp("2018-09-13")
TEST_START = pd.Timestamp("2018-10-20")


def load_data(path=DATASET):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y %H:%M", errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Expected unique, nonmissing interval timestamps.")
    return frame


def labels(frame):
    ids = pd.Categorical(frame[TARGET], categories=CLASSES).codes
    if (ids < 0).any():
        raise ValueError("Unknown or missing Load_Type.")
    return ids.astype(int)


def split_data(frame, method, seed=42):
    if method == "random":
        development, test = train_test_split(
            frame, test_size=0.20, random_state=seed, stratify=frame[TARGET]
        )
        train, validation = train_test_split(
            development, test_size=0.125, random_state=seed, stratify=development[TARGET]
        )
    elif method == "chronological":
        ordered = frame.sort_values("date")
        dates = ordered["date"]
        train = ordered[dates < VALIDATION_START]
        validation = ordered[(dates >= VALIDATION_START) & (dates < TEST_START)]
        test = ordered[dates >= TEST_START]
    else:
        raise ValueError(f"Unknown split method: {method}")
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Train, validation, and test must each contain observations.")
    return train, validation, test


def feature_frame(frame, feature_set):
    features = frame.loc[:, list(FEATURE_SETS[feature_set])].copy()
    if feature_set == "electrical-only":
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Numeric features must be finite.")
        return features
    # Derive schedule inputs from the timestamp alone, including at midnight.
    dates = frame["date"].dt
    seconds = dates.hour * 3600 + dates.minute * 60 + dates.second
    angle = 2 * np.pi * seconds / 86400
    features["NSM_sin"] = np.sin(angle)
    features["NSM_cos"] = np.cos(angle)
    features["WeekStatus"] = np.where(dates.dayofweek < 5, "Weekday", "Weekend")
    features["Day_of_week"] = dates.day_name()
    if not np.isfinite(features.drop(columns=list(CATEGORICAL)).to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite.")
    return features


def make_preprocessor(feature_set):
    numbers = [*FEATURE_SETS[feature_set], "NSM_sin", "NSM_cos"]
    if feature_set == "electrical-only":
        return ColumnTransformer([("numeric", MinMaxScaler(), list(ELECTRICAL))])
    return ColumnTransformer(
        [
            ("numeric", MinMaxScaler(), numbers),
            ("calendar_categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(CATEGORICAL)),
        ]
    )


def prepare_development(train, validation, feature_set):
    preprocessor = make_preprocessor(feature_set)
    x_train = preprocessor.fit_transform(feature_frame(train, feature_set))
    x_val = preprocessor.transform(feature_frame(validation, feature_set))
    return preprocessor, x_train, x_val

