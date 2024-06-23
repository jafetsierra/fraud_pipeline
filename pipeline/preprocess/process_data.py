import logging
import typer
import pandas as pd
from typer import Option
from typing_extensions import Annotated
from pipeline.preprocess.utils import prepare_data

app = typer.Typer()

@app.command()
def main(
    train_csv: Annotated[str, Option("--train-path")],
    test_csv: Annotated[str, Option("--test-path")],
    label_column: Annotated[str, Option("--label-column")],
    train_data_path: Annotated[str, Option("--train-data-path")],
    test_data_path: Annotated[str, Option("--test-data-path")],
):
    """Process the data for training."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Prepare the training data
    X_train, y_train, _ = prepare_data(train_df, label_column)
    X_test, y_test, _ = prepare_data(test_df, label_column)

    # Save the processed data
    pd.DataFrame(X_train).to_csv(f"{train_data_path}_features.csv", index=False)
    y_train.to_csv(f"{train_data_path}_labels.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{test_data_path}_features.csv", index=False)
    y_test.to_csv(f"{test_data_path}_labels.csv", index=False)

    logging.info("Training data saved to %s", train_data_path)
    logging.info("Test data saved to %s", test_data_path)

if __name__ == "__main__":
    app()
