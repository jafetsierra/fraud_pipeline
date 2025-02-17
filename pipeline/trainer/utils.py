"""Utility functions for the training pipeline."""

import json
import logging

from cloudpathlib import AnyPath
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_config(config_path: AnyPath):
    """Load model configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding="uft-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found at path: %s", config_path)
        raise
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from configuration file at path: %s", config_path)
        raise

model_classes = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier
}
