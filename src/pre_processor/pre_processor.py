from typing import Tuple, Union

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src import config


class PreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, data: pd.DataFrame, production: bool = False):
        self.data = data
        self.production = production

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """transform the data by encoding categorical variables and imputing missing values.

        Args:
            X (pd.Data): features
            y (pd.Series, optional): target. Defaults to None.
        """
        self.preprocess_data()
        if self.production:
            X_train, y_train = self.get_data_split()
            X_train, y_train = self.oversample(X_train, y_train)
            return X_train, y_train
        else:
            X_train, X_test, y_train, y_test = self.get_data_split()
            X_train, y_train = self.oversample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def preprocess_data(self) -> None:
        """
        pre process data by encoding categorical variables and imputing missing values.
        """
        encoded = pd.get_dummies(self.data[config.CATEGORICAL_COLUMNS], prefix=config.CATEGORICAL_COLUMNS)

        # Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(config.CATEGORICAL_COLUMNS, axis=1, inplace=True)

        # Impute missing values of BMI & drop ID
        self.data.bmi = self.data.bmi.fillna(0)
        self.data.drop([config.ID_COLUMN], axis=1, inplace=True)

    def get_data_split(
        self,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Splits the dataset into training and testing sets if not in production mode,
        otherwise returns the features and target variables without splitting.

        Returns:
            Union[Tuple[DataFrame, DataFrame], Tuple[DataFrame, DataFrame, DataFrame, DataFrame]]:
            - If in production mode: A tuple containing the features (X) and target (y) as DataFrames.
            - If not in production mode: A tuple containing the training features (X_train), testing features (X_test),
              training target (y_train), and testing target (y_test) as DataFrames.
        """

        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        if self.production:
            return X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test

    def oversample(self, X_train, y_train) -> Tuple[pd.DataFrame, pd.Series]:
        """Oversample the minority class in the target variable.

        Args:
            X_train (_type_): features
            y_train (_type_): target variable

        Returns:
            Tuple[pd.DataFrame, pd.Series]: oversampled features and target variables
        """
        oversampler = RandomOverSampler(sampling_strategy="minority")

        features_np_array = X_train.to_numpy()
        target_np_array = y_train.to_numpy()

        features_np_array, target_np_array = oversampler.fit_resample(features_np_array, target_np_array)

        features_oversampled = pd.DataFrame(features_np_array, columns=X_train.columns)
        target_oversampled = pd.Series(target_np_array, name=y_train.name)

        return features_oversampled, target_oversampled
