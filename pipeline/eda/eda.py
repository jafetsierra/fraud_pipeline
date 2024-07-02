# pylint: disable=line-too-long
"""
Exploratory Data Analysis (EDA) Script

This module performs exploratory data analysis on a given dataset and logs the results to Weights & Biases (wandb).

The script reads a CSV file, processes the data, generates various plots and statistical summaries,
and saves the results both locally and to a wandb project.

Functions:
    eda: Main function to perform exploratory data analysis.

Usage:
    Run this script from the command line with the required arguments:
    python eda.py --input-path <path_to_input_csv> --output-path <path_to_save_plots> --wandb-project <wandb_project_name>

Dependencies:
    - pandas
    - seaborn
    - matplotlib
    - typer
    - cloudpathlib
    - wandb
    - numpy

Note:
    This script assumes that the input CSV file has a specific structure and certain columns.
    Make sure to review and modify the script if your data structure differs.
"""

from typing import Annotated

import typer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from cloudpathlib import AnyPath
import wandb

# pylint: disable=too-many-locals
def eda(
    input_path: Annotated[str, typer.Option("--input-path", help="Path to the input data")],
    output_path: Annotated[str, typer.Option("--output-path", help="Path to save the output plots")],
    wandb_project: Annotated[str, typer.Option("--wandb-project", help="wandb project name")],
):
    """Perform exploratory data analysis."""

    wandb.init(project=wandb_project, job_type="eda")

    df = pd.read_csv(input_path)

    df.pop('Unnamed: 0')

    description = df.describe()
    description_table = wandb.Table(columns=description.columns,dataframe=description.reset_index())
    wandb.log({"description_table": description_table})

    output_dir = AnyPath(output_path)
    # pylint: disable=no-member
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_columns = ['Unnamed: 0','trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']

    numerical_features = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns).tolist()
    #categorical_features = df.select_dtypes(exclude=[np.number]).columns.difference(exclude_columns).tolist()

    if numerical_features:
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(df[numerical_features].corr(), annot=True, fmt='.2f')
        heatmap_path = output_dir / 'correlation_heatmap.png'
        heatmap.figure.savefig(heatmap_path)
        plt.close()
        wandb.log({"correlation_heatmap": wandb.Image(str(heatmap_path))})

    for column in numerical_features:
        plt.figure(figsize=(10, 6))
        distribution_plot = sns.histplot(df[column], kde=True, bins=100, stat='probability')
        plot_path = output_dir / f'distribution_{column}.png'
        distribution_plot.figure.savefig(plot_path)
        plt.close()
        wandb.log({f"distribution_{column}": wandb.Image(str(plot_path))})

    for column in numerical_features:
        plt.figure(figsize=(10, 6))
        boxplot = sns.boxplot(x=df[column])
        plot_path = output_dir / f'boxplot_{column}.png'
        boxplot.figure.savefig(plot_path)
        plt.close()
        wandb.log({f"boxplot_{column}": wandb.Image(str(plot_path))})

    if 'is_fraud' in df.columns:
        fraud_distribution_plot = sns.countplot(x='is_fraud', data=df)
        fraud_distribution_path = output_dir / 'fraud_distribution.png'
        fraud_distribution_plot.figure.savefig(fraud_distribution_path)
        plt.close()
        wandb.log({"fraud_distribution": wandb.Image(str(fraud_distribution_path))})

        target_correlation = df[numerical_features].corr()['is_fraud'].sort_values(ascending=False)
        target_correlation.name = "correlation_table"
        target_correlation_table = wandb.Table(columns=['correlation_with_is_fraud'],dataframe=target_correlation.reset_index())
        wandb.log({"correlation_table": target_correlation_table})

if __name__ == "__main__":
    typer.run(eda)
