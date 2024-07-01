# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typer
from typing import Annotated
from cloudpathlib import AnyPath
import wandb
import numpy as np

def eda(
    input_path: Annotated[str, typer.Option("--input-path", help="Path to the input data")],
    output_path: Annotated[str, typer.Option("--output-path", help="Path to save the output plots")],
    wandb_project: Annotated[str, typer.Option("--wandb-project", help="wandb project name")],
):
    """Perform exploratory data analysis."""

    wandb.init(project=wandb_project, job_type="eda")

    df = pd.read_csv(input_path)

    df.pop('Unnamed: 0')

    info = df.info()
    description = df.describe()
    description_table = wandb.Table(columns=description.columns,dataframe=description.reset_index())
    wandb.log({"description_table": description_table})

    output_dir = AnyPath(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_columns = ['Unnamed: 0','trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']

    numerical_features = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns).tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.difference(exclude_columns).tolist()

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
        fraud_counts = df['is_fraud'].value_counts()
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
