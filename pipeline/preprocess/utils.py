""" This module contains utility functions for preprocessing data. """
from typing import Tuple
import pandas as pd

def prepare_data(train_df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare the data for training."""
    x_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    return x_train, y_train
