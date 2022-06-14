from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.config import settings

# impute_cols if empty, will be interpreted as we want to impute for all the feature columns.


def simple_impute(df, cols, strategy, missing=np.NaN):
    orig_count = len(df)
    logger.info(f"Initial data count: {orig_count:,}")

    if strategy is None:
        logger.warning(
            f"Specified strategy = {str(strategy)}. No imputation is applied"
        )
    else:
        logger.warning(f"Specified strategy = {str(strategy)}. Imputing values...")
        imputer = SimpleImputer(missing_values=missing, strategy=strategy)

        for col in cols:
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))

    logger.info(f"Final data count: {len(df):,}")

    return df


def drop_nulls(df, cols):
    logger.info(f"Retaining the following columns: {cols}")
    reduced_df = df.loc[:, cols]
    orig_count = len(reduced_df)

    if reduced_df.isnull().values.any() > 0:
        logger.warning(f"Removing any null rows. Initial data count: {orig_count:,}")

        reduced_df.dropna(how="any", inplace=True)
        reduced_df.reset_index(drop=True, inplace=True)

        null_rows = orig_count - len(reduced_df)

        logger.warning(
            f"Dropped {null_rows:,} rows with nulls. Final data count: {len(reduced_df):,}"
        )
    else:
        logger.warning(f"No null rows. Initial data count: {orig_count:,}")

    return reduced_df


def balance_data(df, label, seed=settings.SEED):
    """Returns a balanced dataset based on set target label
    Args:
        df (dataframe): A dataframe for training
        label (string): target variable for oversampling
        seed (int): random seed for sampling
    Returns:
        dataframe: balanced df
    """
    logger.info(f"Balancing data (target col = {label})")
    # Generate X and y
    X = df.drop(columns=[label])
    y = df[label].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Oversampling
    oversample = SMOTE(random_state=seed)
    X_os, y_os = oversample.fit_resample(X, y)

    # summarize distribution
    counter = Counter(y_os)
    for k, v in counter.items():
        per = v / len(y) * 100
        print(f"Class={k}, n={v} ({per:.3f}%)")

    balanced_df = X_os.copy()
    balanced_df[label] = encoder.inverse_transform(y_os)

    return balanced_df
