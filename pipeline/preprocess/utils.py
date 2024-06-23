import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

def prepare_data(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Prepare the data for training."""

    # Separate features and label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Handling missing values
    X.fillna(method='ffill', inplace=True)

    # Define categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

    # Define preprocessing for numerical features: scaling
    numerical_transformer = StandardScaler()

    # Define preprocessing for categorical features: one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Define the preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # Fit and transform the features using the pipeline
    X_preprocessed = preprocessing_pipeline.fit_transform(X)

    return X_preprocessed, y, preprocessing_pipeline
