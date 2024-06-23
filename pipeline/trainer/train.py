import logging
import typer
import json
from typer import Option
from typing_extensions import Annotated
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pipeline.preprocess.dataset import TransactionDataset
from pipeline.trainer.sklearn_trainer import SklearnTrainer

app = typer.Typer()

# Dictionary of available models
MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier
}

def load_model(model_name: str, model_params: dict):
    model_class = MODEL_CLASSES.get(model_name)
    if model_class:
        return model_class(**model_params)
    raise ValueError(f"Model {model_name} is not supported.")

@app.command()
def main(
    config_path: Annotated[str, Option("--config-path", help="Path to the configuration file")],
    train_data_path: Annotated[str, Option("--train-data-path", help="Path to the training data")],
    test_data_path: Annotated[str, Option("--test-data-path", help="Path to the test data")],
    cv: Annotated[int, Option("--cv", help="Number of cross-validation folds")],
):
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load the processed data
    train_data = TransactionDataset.load(f"{train_data_path}_features.csv", f"{train_data_path}_labels.csv")
    test_data = TransactionDataset.load(f"{test_data_path}_features.csv", f"{test_data_path}_labels.csv")

    # Set up baseline model
    baseline_model = load_model(config["baseline_model"]["name"], config["baseline_model"]["parameters"])
    steps = [("classifier", baseline_model)]

    baseline_trainer = SklearnTrainer(steps)

    # Perform cross-validation for baseline model
    baseline_cv_score = baseline_trainer.cross_validate(train_data, cv=cv)
    logging.info(f"Baseline model cross-validation mean accuracy: {baseline_cv_score}")

    # Train baseline model
    baseline_trainer.train(train_data)

    # Evaluate baseline model
    baseline_accuracy = baseline_trainer.evaluate(test_data)
    logging.info(f"Baseline model test accuracy: {baseline_accuracy}")

    # Save the baseline model
    baseline_trainer.save_model('path/to/save/baseline_model.pkl')

    # Set up production model
    production_model = load_model(config["production_model"]["name"], config["production_model"]["parameters"])
    steps = [("classifier", production_model)]

    production_trainer = SklearnTrainer(steps)

    # Perform cross-validation for production model
    production_cv_score = production_trainer.cross_validate(train_data, cv=cv)
    logging.info(f"Production model cross-validation mean accuracy: {production_cv_score}")

    # Train production model
    production_trainer.train(train_data)

    # Evaluate production model
    production_accuracy = production_trainer.evaluate(test_data)
    logging.info(f"Production model test accuracy: {production_accuracy}")

    # Save the production model
    production_trainer.save_model('path/to/save/production_model.pkl')

if __name__ == "__main__":
    app()
