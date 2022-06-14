import joblib
import numpy as np
from loguru import logger

from src.data_processing import feature_collection_pipeline


def predict(
    locations_df,
    start_date,
    end_date,
    id_col,
    hrsl_tif,
    model_path,
    bbox_size_km=1,
    pred_col="predicted_pm2.5",
):

    logger.info(
        f"Running prediction on {len(locations_df):,} locations from {start_date} to {end_date}..."
    )

    # Create base DF from the locations (collect the necessary features)
    logger.info("Collecting features...")
    base_df = feature_collection_pipeline.collect_features_for_locations(
        locations_df=locations_df,
        start_date=start_date,
        end_date=end_date,
        id_col=id_col,
        hrsl_tif=hrsl_tif,
        bbox_size_km=bbox_size_km,
        # Customized the list of GEE datasets because the latest model doesn't use MAIAC
        gee_datasets=[
            feature_collection_pipeline.S5P_AAI_CONFIG,
            feature_collection_pipeline.CAMS_AOD_CONFIG,
            feature_collection_pipeline.NDVI_CONFIG,
            feature_collection_pipeline.ERA5_CONFIG,
        ],
    )

    logger.info("Running the model...")
    base_df[pred_col] = predict_directly(base_df, model_path)

    return base_df


def predict_directly(data_df, model_path):
    # Load Model
    model = joblib.load(model_path)

    # Filter to only the relevant columns
    keep_cols = model.feature_names  # This was saved from the train script
    ml_df = data_df[keep_cols]

    # Print out % missing values per feature
    for keep_col in keep_cols:
        pct_missing = len(data_df[data_df[keep_col].isna()]) / len(data_df)
        logger.debug(f"{keep_col} missing = {pct_missing:.2%}")

    # Run model
    preds = model.predict(ml_df)

    # Post-process to avoid negative values
    preds = np.clip(preds, 0, 500)

    return preds
