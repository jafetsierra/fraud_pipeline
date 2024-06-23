from abc import ABC, abstractmethod
from pipeline.preprocess.dataset import TransactionDataset

class BaseTrainer(ABC):
    @abstractmethod
    def cross_validate(self, train_data: TransactionDataset, cv: int) -> float:
        """Perform cross-validation on the training set."""
        pass

    @abstractmethod
    def train(self, train_data: TransactionDataset) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, test_data: TransactionDataset) -> float:
        """Evaluate the model on the test set."""
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file."""
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load the trained model from a file."""
        pass
