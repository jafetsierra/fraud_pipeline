""" This script is responsible for processing the data before training the model."""
import logging
import typer
import pandas as pd
from typer import Option
from typing_extensions import Annotated
from .dataset import TransactionDataset
from .utils import prepare_data

app = typer.Typer()

@app.command()
def main(
    train_csv: Annotated[str, Option("--train-path")],
    test_csv: Annotated[str, Option("--test-path")],
    label_column: Annotated[str, Option("--label-column")],
):
    """Process the data for training."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Prepare the training data
    x_train, y_train = prepare_data(train_df, label_column)
    x_test, y_test = prepare_data(test_df, label_column)

    # Create dataclass instances
    train_data = TransactionDataset(features=x_train, labels=y_train)
    test_data = TransactionDataset(features=x_test, labels=y_test)

    # Output for confirmation (this could be replaced with saving to files, logging, etc.)
    logging.info("Training data: %s", train_data)
    logging.info("Test data: %s", test_data)

if __name__ == "__main__":
    app()
