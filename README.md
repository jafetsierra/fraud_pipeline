# Fraud Detection Pipeline

This repository contains a comprehensive pipeline for fraud detection, including Exploratory Data Analysis (EDA), data preparation, model selection, model training, and evaluation modules.

## Overview

The fraud detection pipeline consists of the following modules:

1. Exploratory Data Analysis (EDA)
2. Data Preparation
3. Model Selection
4. Model Training
5. Model Evaluation

Each module can be run independently or as part of the complete pipeline.

## Requirements

- Python 3.10+
- Poetry (for dependency management)

## Installation

1. Clone this repository:
   ```
   git clone <repository_url>
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```


## Modules

### 1. Exploratory Data Analysis (EDA)

The EDA module performs initial analysis on the input dataset to gain insights and prepare for model development.

#### Features

- Reads CSV data
- Generates descriptive statistics
- Creates correlation heatmaps for numerical features
- Plots distribution and box plots for numerical features
- Analyzes fraud distribution
- Logs all results to wandb for easy tracking and visualization

#### Usage

Run the EDA script from the command line:

```
python eda.py --input-path <path_to_input_csv> --output-path <path_to_save_plots> --wandb-project <wandb_project_name>
```

Arguments:
- `--input-path`: Path to the input CSV file
- `--output-path`: Directory to save the output plots
- `--wandb-project`: Name of the wandb project to log results

#### Output

The EDA module generates:
1. Descriptive statistics table (logged to wandb)
2. Correlation heatmap for numerical features
3. Distribution plots for each numerical feature
4. Box plots for each numerical feature
5. Fraud distribution plot
6. Target correlation table

All plots are saved locally in the specified output directory and logged to the wandb project.

### 2. Data Preparation
The Data Preparation module preprocesses the input data for both training and testing sets, preparing them for model training.
Features

Loads training and test data from CSV files
Fills missing values using forward fill method
Converts transaction date/time to datetime format
Extracts hour, day, and month from the transaction datetime
Selects relevant features for the model
Scales the features using StandardScaler
Saves the prepared data as CSV files

Usage
Run the Data Preparation script from the command line:
Copypoetry run python data_preparation.py --train-input-path <path_to_train_csv> --test-input-path <path_to_test_csv> --output-path <path_to_save_prepared_data>
Arguments:

--train-input-path: Path to the training data CSV file
--test-input-path: Path to the test data CSV file
--output-path: Directory to save the prepared data files

Output
The Data Preparation module generates the following files in the specified output directory:

X_train.csv: Scaled features for the training set
X_test.csv: Scaled features for the test set
y_train.csv: Target variable for the training set
y_test.csv: Target variable for the test set

### 3. Model Selection

The Model Selection module compares different machine learning models and their hyperparameters to select the best model for fraud detection.

#### Features

- Loads training data and model configuration
- Initializes models based on the configuration
- Performs grid search with cross-validation for each model
- Evaluates models using precision, recall, and F1 score
- Selects the best model based on F1 score
- Saves the best model configuration
- Generates and saves model comparison plots and metrics
- Logs results to Weights & Biases (wandb)

#### Usage

Run the Model Selection script from the command line:

```
poetry run python model_selection.py --train-data-path <path_to_train_data> --config-path <path_to_config> --output-path <path_to_save_output> --wandb-project <wandb_project_name>
```

Arguments:
- `--train-data-path`: Path to the directory containing training data (X_train.csv and y_train.csv)
- `--config-path`: Path to the model selection configuration file
- `--output-path`: Directory to save the output files
- `--wandb-project`: Name of the Weights & Biases project for logging

#### Output

The Model Selection module generates the following outputs:
1. `training.json`: Configuration file for the best selected model
2. `model_comparison.csv`: CSV file containing performance metrics for all models
3. `model_comparison.png`: Bar plot comparing model performances
4. Logs model comparison results to Weights & Biases

### 4. Model Training

The Model Training module trains the selected model using the prepared data and the configuration from the model selection phase.

#### Features

- Loads training data
- Loads model configuration
- Initializes the selected model with the best parameters
- Trains the model on the training data
- Logs training metrics and plots to Weights & Biases (wandb)
- Saves the trained model
- Saves and logs the final model parameters

#### Usage

Run the Model Training script from the command line:

```
poetry run python training.py --train-data-path <path_to_train_data> --config-path <path_to_config> --output-path <path_to_save_output> --wandb-project <wandb_project_name>
```

Arguments:
- `--train-data-path`: Path to the directory containing training data (X_train.csv and y_train.csv)
- `--config-path`: Path to the model training configuration file
- `--output-path`: Directory to save the output files
- `--wandb-project`: Name of the Weights & Biases project for logging

#### Output

The Model Training module generates the following outputs:
1. `trained_model.pkl`: Saved trained model file
2. `final_model_parameters.json`: JSON file containing the final model parameters
3. Logs training metrics, plots, and artifacts to Weights & Biases

### 5. Model Evaluation

The Model Evaluation module assesses the performance of the trained model using the test dataset.

#### Features

- Loads test data
- Loads the trained model
- Makes predictions on the test data
- Calculates evaluation metrics (precision, recall, F1 score, ROC AUC)
- Generates a classification report
- Logs evaluation metrics and the classification report to Weights & Biases (wandb)

#### Usage

Run the Model Evaluation script from the command line:

```
poetry run python evaluation.py --test-data-path <path_to_test_data> --model-path <path_to_trained_model> --wandb-project <wandb_project_name>
```

Arguments:
- `--test-data-path`: Path to the directory containing test data (X_test.csv and y_test.csv)
- `--model-path`: Path to the trained model file
- `--wandb-project`: Name of the Weights & Biases project for logging

#### Output

The Model Evaluation module generates the following outputs:
1. Logs evaluation metrics (precision, recall, F1 score, ROC AUC) to Weights & Biases
2. Logs the classification report to Weights & Biases
3. Prints the classification report to the console

## Complete Pipeline Usage

To run the complete fraud detection pipeline, execute the following steps in order:

1. Exploratory Data Analysis:
   ```
   poetry run python eda.py --input-path <path_to_input_data> --output-path <path_to_save_plots> --wandb-project <wandb_project_name>
   ```

2. Data Preparation:
   ```
   poetry run python data_preparation.py --train-input-path <path_to_train_csv> --test-input-path <path_to_test_csv> --output-path <path_to_save_prepared_data>
   ```

3. Model Selection:
   ```
   poetry run python model_selection.py --train-data-path <path_to_train_data> --config-path <path_to_config> --output-path <path_to_save_output> --wandb-project <wandb_project_name>
   ```

4. Model Training:
   ```
   poetry run python training.py --train-data-path <path_to_train_data> --config-path <path_to_config> --output-path <path_to_save_output> --wandb-project <wandb_project_name>
   ```

5. Model Evaluation:
   ```
   poetry run python evaluation.py --test-data-path <path_to_test_data> --model-path <path_to_trained_model> --wandb-project <wandb_project_name>
   ```

Ensure that you provide the correct paths and Weights & Biases project name for each step.


## License

[None] since this is just a small test for bold