# pylint: disable=line-too-long
"""
Model Selection Module for Fraud Detection

This module performs model selection for the fraud detection pipeline by comparing
different machine learning models and their hyperparameters.

Functions:
    model_selection: Main function to select the best model for fraud detection.

The script performs the following operations:
1. Loads training data and model configuration
2. Initializes models based on the configuration
3. Performs grid search with cross-validation for each model
4. Evaluates models using precision, recall, and F1 score
5. Selects the best model based on F1 score
6. Saves the best model configuration
7. Generates and saves model comparison plots and metrics
8. Logs results to Weights & Biases (wandb)

Usage:
    Run this script from the command line with the required arguments:
    python model_selection.py --train-data-path <path_to_train_data> --config-path <path_to_config> --output-path <path_to_save_output> --wandb-project <wandb_project_name>

Dependencies:
    - pandas
    - scikit-learn
    - typer
    - cloudpathlib
    - wandb
    - matplotlib

Note:
    This script assumes a specific structure for the input data and configuration files.
    Modify the script if your data or config structure differs.
"""


import json
import logging
from typing import Annotated

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import typer

from cloudpathlib import AnyPath
import matplotlib.pyplot as plt
import wandb

from config import CONFIG_DIR
from .utils import load_model_config, model_classes


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=too-many-statements, too-many-locals
def model_selection(
    train_data_path: Annotated[str, typer.Option("--train-data-path", help="Path to the training data")],
    config_path: Annotated[str, typer.Option("--config-path", help="Path to the model selection config")],
    output_path: Annotated[str, typer.Option("--output-path", help="Path to save the best model config")],
    wandb_project: Annotated[str, typer.Option("--wandb-project", help="WandB project name")],
):
    """Select the best model for fraud detection."""

    try:
        wandb.init(project=wandb_project, job_type="model_selection")

        x_train = pd.read_csv(AnyPath(train_data_path) / 'x_train.csv')
        y_train = pd.read_csv(AnyPath(train_data_path) / 'y_train.csv').values.ravel()

        config = load_model_config(AnyPath(config_path))

        models = {}
        param_grids = {}
        for model_name, model_info in config["models"].items():
            try:
                model_class = model_classes[model_name]
                models[model_name] = model_class(**model_info["parameters"])
                param_grids[model_name] = model_info["grid_search"]
            except KeyError:
                logger.error("Model %s is not supported.", model_name)
                raise

        best_models = {}
        model_metrics = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            clf = GridSearchCV(model, param_grids[name], cv=skf, scoring='f1')
            clf.fit(x_train, y_train)
            best_model = clf.best_estimator_
            best_models[name] = best_model

            precision = cross_val_score(best_model, x_train, y_train, cv=skf, scoring=make_scorer(precision_score,average='macro')).mean()
            recall = cross_val_score(best_model, x_train, y_train, cv=skf, scoring=make_scorer(recall_score,average='macro')).mean()
            f1 = cross_val_score(best_model, x_train, y_train, cv=skf, scoring=make_scorer(f1_score,average='macro')).mean()


            model_metrics[name] = {'precision': precision, 'recall': recall, 'f1': f1}

        best_model_name = max(model_metrics, key=lambda name: model_metrics[name]['f1'])
        best_model = best_models[best_model_name]

        logger.info("Best model: %s", best_model_name)

        output_dir = AnyPath(output_path)
        # pylint: disable=no-member
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_config = {
            "model": {
                "name": best_model_name,
                "parameters": best_model.get_params()
            }
        }
        best_model_config_path = CONFIG_DIR / 'training.json'
        with open(best_model_config_path, 'w', encoding="uft-8") as f:
            json.dump(best_model_config, f)
        logger.info("Best model parameters saved at: %s", best_model_config_path)

        model_comparison_df = pd.DataFrame(model_metrics).T

        model_comparison_df.to_csv(output_dir / 'model_comparison.csv', index=True)

        model_comparison_df.plot(kind='bar', figsize=(12, 8))
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Comparison - Precision, Recall, F1')
        plt.legend(loc='upper right')

        comparison_plot_path = output_dir / 'model_comparison.png'
        plt.savefig(comparison_plot_path)
        plt.close()

        wandb.log({"model_comparison_table": wandb.Table(columns=['precision','recall','f1'],dataframe=model_comparison_df)})
        wandb.log({"model_comparison_plot": wandb.Image(str(comparison_plot_path))})

    except Exception as error:
        logger.error("An error occurred during model selection: %s",error)
        raise

if __name__ == "__main__":
    typer.run(model_selection)
