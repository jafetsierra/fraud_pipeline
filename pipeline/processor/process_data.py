import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer
from typing import Annotated
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

    X_train = df_train[features]
    y_train = df_train[target]

    X_test = df_test[features]
    y_test = df_test[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    output_dir = AnyPath(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=features).to_csv(output_dir / 'X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=features).to_csv(output_dir / 'X_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False)
    y_test.to_csv(output_dir / 'y_test.csv', index=False)

if __name__ == "__main__":
    typer.run(data_preparation)
