"""Some helper classes for the cross validation."""
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pydantic import BaseModel

DATE_COLUMN: Literal["date"] = "date"
TARGET: Literal["sales"] = "sales"


class Feature(BaseModel):
    name: str

    @property
    def max_horizon(self) -> int | None:
        """Infer the maximum horizon for this feature from its name.
        For example, the feature "lag_1___max_horizon_1" ("yesterday's" sales),
        can only be used by horizon=1 models.
        """
        if "___max_horizon_" not in self.name:
            return None
        return int(self.name.split("___max_horizon_")[1])

    def is_available(self, *, horizon: int) -> bool:
        if self.max_horizon is None or horizon <= self.max_horizon:
            return True
        return False


class Features(BaseModel):
    feature_names: list[str]

    @property
    def features(self) -> list[Feature]:
        return [Feature(name=feature_name) for feature_name in self.feature_names]

    def get_unavailable_features(self, horizon: int) -> list[str]:
        return [feature.name for feature in self.features if not feature.is_available(horizon=horizon)]

    def get_available_features_for_horizon(self, horizon: int) -> list[str]:
        return [feature.name for feature in self.features if feature.is_available(horizon=horizon)]

    def get_available_features(self, horizons: list[int]) -> list[str]:
        features = list(
            set([feat for horizon in horizons for feat in self.get_available_features_for_horizon(horizon=horizon)])
        )
        features.sort()
        return features


class Data:
    def __init__(self, data: pd.DataFrame, features: Features, horizons: list[int]) -> None:
        self.horizons = horizons
        self.features = features
        self.data = self._prep_data(data)

    @property
    def available_features(self) -> list[str]:
        return self.features.get_available_features(horizons=self.horizons)

    def get_train_data(self, forecast_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
        is_train = self.data.index <= forecast_date
        y_train = self.data.loc[is_train, TARGET]
        X_train = self.data.loc[is_train, self.available_features]
        return X_train, y_train

    def get_test_data(self, forecast_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
        test_dates = [forecast_date + pd.Timedelta(days=horizon) for horizon in self.horizons]
        is_test = self.data.index.isin(test_dates)
        y_test = self.data.loc[is_test, TARGET]
        X_test = self.data.loc[is_test, self.available_features]
        return X_test, y_test

    def _prep_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[self.available_features + [TARGET]]


class MaskedData(Data):
    def get_test_data(self, forecast_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
        is_test = None

        for horizon in self.horizons:
            is_test_date = self.data.index == (forecast_date + pd.Timedelta(days=horizon))
            is_horizon = self.data["horizon"] == horizon
            if is_test is None:
                is_test = is_test_date & is_horizon
            else:
                is_test |= is_test_date & is_horizon

        y_test = self.data.loc[is_test, TARGET]
        X_test = self.data.loc[is_test, self.available_features + ["horizon"]].copy()

        for horizon in self.horizons:
            features_to_mask_ = self.features.get_unavailable_features(horizon=horizon)
            X_test.loc[lambda df: df["horizon"] == horizon, features_to_mask_] = np.nan

        return X_test.drop(columns="horizon"), y_test

    def _prep_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            data.assign(horizon=horizon).pipe(self._mask_features, horizon=horizon) for horizon in self.horizons
        ).sort_index()

    def _mask_features(self, data: pd.DataFrame, horizon: int, p: float = 1 / 3) -> pd.DataFrame:
        features_to_mask = self.features.get_unavailable_features(horizon=horizon)
        n = len(data)
        data.loc[np.random.choice([True, False], size=n, p=(p, 1 - p)), features_to_mask] = np.nan
        return data


class Model(BaseModel):
    model: LGBMRegressor
    data: Data

    @property
    def horizons(self) -> list[int]:
        return self.data.horizons

    class Config:
        arbitrary_types_allowed = True
