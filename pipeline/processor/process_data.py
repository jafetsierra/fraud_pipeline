# pylint: disable=line-too-long
"""
Data Preparation Module for Fraud Detection

This module prepares the data for the fraud detection model by performing
various preprocessing steps on both training and test datasets.

Functions:
    data_preparation: Main function to prepare data for training.

The script performs the following operations:
1. Loads training and test data from CSV files
2. Fills missing values using forward fill method
3. Converts transaction date/time to datetime format
4. Extracts hour, day, and month from the transaction datetime
5. Selects relevant features for the model
6. Scales the features using StandardScaler
7. Saves the prepared data (x_train, x_test, y_train, y_test) as CSV files

Usage:
    Run this script from the command line with the required arguments:
    python data_preparation.py --train-input-path <path_to_train_csv> --test-input-path <path_to_test_csv> --output-path <path_to_save_prepared_data>

Dependencies:
    - pandas
    - scikit-learn
    - typer
    - cloudpathlib

Note:
    This script assumes specific column names and data structure in the input CSV files.
    Modify the script if your data structure differs.
"""

from typing import Annotated

import typer
import pandas as pd

from sklearn.preprocessing import StandardScaler
from cloudpathlib import AnyPath

def data_preparation(
    train_input_path: Annotated[str, typer.Option("--train-input-path", help="Path to the training data")],
    test_input_path: Annotated[str, typer.Option("--test-input-path", help="Path to the test data")],
    output_path: Annotated[str, typer.Option("--output-path", help="Path to save the prepared data")],
):
    """Prepare data for training."""

    df_train = pd.read_csv(train_input_path)
    df_test = pd.read_csv(test_input_path)

    df_train = df_train.ffill()
    df_train = df_test.ffill()

    df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])
    df_test['trans_date_trans_time'] = pd.to_datetime(df_test['trans_date_trans_time'])

    df_train['hour'] = df_train['trans_date_trans_time'].dt.hour
    df_train['day'] = df_train['trans_date_trans_time'].dt.day
    df_train['month'] = df_train['trans_date_trans_time'].dt.month

    df_test['hour'] = df_test['trans_date_trans_time'].dt.hour
    df_test['day'] = df_test['trans_date_trans_time'].dt.day
    df_test['month'] = df_test['trans_date_trans_time'].dt.month

    features = ['amt', 'city_pop', 'hour', 'day', 'month', 'lat', 'long', 'merch_lat', 'merch_long']
    target = 'is_fraud'

    x_train = df_train[features]
    y_train = df_train[target]

    x_test = df_test[features]
    y_test = df_test[target]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    output_dir = AnyPath(output_path)
    # pylint: disable=no-member
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(x_train_scaled, columns=features).to_csv(output_dir / 'x_train.csv', index=False)
    pd.DataFrame(x_test_scaled, columns=features).to_csv(output_dir / 'x_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False)
    y_test.to_csv(output_dir / 'y_test.csv', index=False)

if __name__ == "__main__":
    typer.run(data_preparation)
