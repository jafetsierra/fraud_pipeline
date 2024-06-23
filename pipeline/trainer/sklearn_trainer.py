from pipeline.preprocess.dataset import TransactionDataset
from pipeline.trainer.base_trainer import BaseTrainer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import pandas as pd

class SklearnTrainer(BaseTrainer, Pipeline):
    def __init__(self, steps: list):
        Pipeline.__init__(self, steps)

    def cross_validate(self, train_data: TransactionDataset, cv: int = 5) -> float:
        """Perform cross-validation on the training set."""
        stratified_k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self, train_data.features, train_data.labels, cv=stratified_k_fold, scoring='accuracy')
        return scores.mean()

    def train(self, train_data: TransactionDataset) -> None:
        """Train the model."""
        self.fit(train_data.features, train_data.labels)

    def evaluate(self, test_data: TransactionDataset) -> float:
        """Evaluate the model on the test set."""
        return self.score(test_data.features, test_data.labels)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file."""
        joblib.dump(self, filepath)

    def load_model(self, filepath: str) -> None:
        """Load the trained model from a file."""
        loaded_model = joblib.load(filepath)
        self.steps = loaded_model.steps
        self.memory = loaded_model.memory

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict labels for the given data."""
        return pd.Series(self.predict(data), index=data.index)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return super().get_params(deep)

    def set_params(self, **params) -> None:
        """Set the parameters of this estimator."""
        return super().set_params(**params)
