import numpy as np
from loguru import logger
from sklearn.impute import SimpleImputer

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
