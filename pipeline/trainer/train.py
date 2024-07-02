# training.py
import pandas as pd
import json
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
