"""
Training Module for Fraud Detection

This module trains the selected model for the fraud detection pipeline using the
prepared data and the configuration from the model selection phase.

Functions:
    train: Main function to train the selected model for fraud detection.

The script performs the following operations:
1. Loads training data
2. Loads model configuration
3. Initializes the selected model with the best parameters
4. Trains the model on the training data
5. Logs training metrics and plots to Weights & Biases (wandb)
6. Saves the trained model
7. Saves and logs the final model parameters

Usage:
    Run this script from the command line with the required arguments:
    python training.py --train-data-path <path_to_train_data> --config-path <path_to_config> --output-path <path_to_save_output> --wandb-project <wandb_project_name>

Dependencies:
    - pandas
    - scikit-learn
    - typer
    - cloudpathlib
    - wandb
    - joblib
    - lightgbm
    - xgboost

Note:
    This script assumes a specific structure for the input data and configuration files.
    Modify the script if your data or config structure differs.
"""

import pandas as pd
import json
import logging

import typer
from pathlib import Path
from typing import Annotated
from cloudpathlib import AnyPath
import joblib
import wandb

from .utils import load_model_config, model_classes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(
    train_data_path: Annotated[str, typer.Option("--train-data-path", help="Path to the training data")],
    config_path: Annotated[str, typer.Option("--config-path", help="Path to the model training config")],
    output_path: Annotated[str, typer.Option("--output-path", help="Path to save the trained model")],
    wandb_project: Annotated[str, typer.Option("--wandb-project", help="WandB project name")],
):
    """Train the selected model."""

    try:
        # Initialize WandB
        wandb.init(project=wandb_project, job_type="training")

        # Load training data
        X_train = pd.read_csv(Path(train_data_path) / 'X_train.csv')
        y_train = pd.read_csv(Path(train_data_path) / 'y_train.csv').values.ravel()

        # Load model configuration
        config = load_model_config(Path(config_path))

        # Extract model and parameters
        model_name = config["model"]["name"]
        model_params = config["model"]["parameters"]

        try:
            model_class = model_classes[model_name]
            model = model_class(**model_params)
        except KeyError:
            logger.error(f"Model {model_name} is not supported.")
            raise

        # Train the model
        model.fit(X_train, y_train)

        # Log model training to WandB
        wandb.sklearn.plot_learning_curve(model, X_train, y_train)
        wandb.sklearn.plot_class_proportions(y_train)
        wandb.log({"parameters": model_params})

        # Save the trained model
        output_dir = AnyPath(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / 'trained_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Model saved at: {model_path}")

        # Log the path to the trained model in WandB
        wandb.log({"model_path": str(model_path)})

        # Save the final model parameters
        final_params_path = output_dir / 'final_model_parameters.json'
        with open(final_params_path, 'w') as f:
            json.dump(model_params, f)
        logger.info(f"Final model parameters saved at: {final_params_path}")

        # Log the final model parameters to WandB
        artifact = wandb.Artifact(name='final_model_parameters', type='parameters', description='Final model parameters', metadata=model_params)
        wandb.log_artifact(artifact)
        wandb.log({"final_model_parameters": artifact.id})

        # Complete WandB run
        wandb.finish()

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise

if __name__ == "__main__":
    typer.run(train)
