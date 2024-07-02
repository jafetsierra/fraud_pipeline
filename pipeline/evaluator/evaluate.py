"""
Evaluation Module for Fraud Detection

This module evaluates the trained model for the fraud detection pipeline using the test data.

Functions:
    evaluate: Main function to evaluate the trained model for fraud detection.

The script performs the following operations:
1. Loads test data
2. Loads the trained model
3. Makes predictions on the test data
4. Calculates evaluation metrics (precision, recall, F1 score, ROC AUC)
5. Generates a classification report
6. Logs evaluation metrics and the classification report to Weights & Biases (wandb)

Usage:
    Run this script from the command line with the required arguments:
    python evaluation.py --test-data-path <path_to_test_data> --model-path <path_to_trained_model> --wandb-project <wandb_project_name>

Dependencies:
    - pandas
    - scikit-learn
    - typer
    - cloudpathlib
    - wandb
    - joblib

Note:
    This script assumes a specific structure for the input data and trained model file.
    Modify the script if your data or model file structure differs.
"""

import pandas as pd
import logging
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import typer
from typing import Annotated
from cloudpathlib import AnyPath
import joblib
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(
    test_data_path: Annotated[str, typer.Option("--test-data-path", help="Path to the test data")],
    model_path: Annotated[str, typer.Option("--model-path", help="Path to the trained model")],
    wandb_project: Annotated[str, typer.Option("--wandb-project", help="WandB project name")],
):
    """Evaluate the trained model."""

    try:
        wandb.init(project=wandb_project, job_type="evaluation")

        X_test = pd.read_csv(AnyPath(test_data_path) / 'X_test.csv')
        y_test = pd.read_csv(AnyPath(test_data_path) / 'y_test.csv')

        model = joblib.load(model_path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        wandb.log({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

        wandb.log({"classification_report": report})

        wandb.finish()

    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        raise

if __name__ == "__main__":
    typer.run(evaluate)
