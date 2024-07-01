# model_selection.py
import pandas as pd
import json
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import typer
from pathlib import Path
from typing import Annotated
from cloudpathlib import AnyPath
import joblib
import wandb
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dictionary to map model names to their classes
model_classes = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier
}

def load_model_config(config_path: Path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at path: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file at path: {config_path}")
        raise

def model_selection(
    train_data_path: Annotated[str, typer.Option("--train-data-path", help="Path to the training data")],
    config_path: Annotated[str, typer.Option("--config-path", help="Path to the model selection config")],
    output_path: Annotated[str, typer.Option("--output-path", help="Path to save the best model")],
    wandb_project: Annotated[str, typer.Option("--wandb-project", help="WandB project name")],
):
    """Select the best model for fraud detection."""

    try:
        # Initialize WandB
        wandb.init(project=wandb_project, job_type="model_selection")

        # Load training data
        X_train = pd.read_csv(Path(train_data_path) / 'X_train.csv')
        y_train = pd.read_csv(Path(train_data_path) / 'y_train.csv').values.ravel()

        # Load model configuration
        config = load_model_config(Path(config_path))

        models = {}
        param_grids = {}
        for model_name, model_info in config["models"].items():
            try:
                model_class = model_classes[model_name]
                models[model_name] = model_class(**model_info["parameters"])
                param_grids[model_name] = model_info["grid_search"]
            except KeyError:
                logger.error(f"Model {model_name} is not supported.")
                raise

        best_models = {}
        model_metrics = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            clf = GridSearchCV(model, param_grids[name], cv=skf, scoring='f1')
            clf.fit(X_train, y_train)
            best_model = clf.best_estimator_
            best_models[name] = best_model

            precision = cross_val_score(best_model, X_train, y_train, cv=skf, scoring=make_scorer(precision_score)).mean()
            recall = cross_val_score(best_model, X_train, y_train, cv=skf, scoring=make_scorer(recall_score)).mean()
            f1 = cross_val_score(best_model, X_train, y_train, cv=skf, scoring=make_scorer(f1_score)).mean()

            model_metrics[name] = {'precision': precision, 'recall': recall, 'f1': f1}

        # Find the best model based on F1 score
        best_model_name = max(model_metrics, key=lambda name: model_metrics[name]['f1'])
        best_model = best_models[best_model_name]

        logger.info(f"Best model: {best_model_name}")

        # Save the best model
        output_dir = AnyPath(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, output_dir / 'best_model.pkl')

        # Create a DataFrame for model comparison
        model_comparison_df = pd.DataFrame(model_metrics).T

        # Save the DataFrame as a CSV file
        model_comparison_df.to_csv(output_dir / 'model_comparison.csv', index=True)

        # Create a bar plot for model comparison
        model_comparison_df.plot(kind='bar', figsize=(12, 8))
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Comparison - Precision, Recall, F1')
        plt.legend(loc='upper right')

        # Save the plot locally
        comparison_plot_path = output_dir / 'model_comparison.png'
        plt.savefig(comparison_plot_path)
        plt.close()

        # Log the table and plot to WandB
        wandb.log({"model_comparison_table": wandb.Table(columns=['precision','recall','f1'],dataframe=model_comparison_df)})
        wandb.log({"model_comparison_plot": wandb.Image(str(comparison_plot_path))})

    except Exception as e:
        logger.error(f"An error occurred during model selection: {e}")
        raise

if __name__ == "__main__":
    typer.run(model_selection)
