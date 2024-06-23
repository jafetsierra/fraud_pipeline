from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class TransactionDataset:
    """A dataset for transaction data."""
    features: Any
    labels: Any

    def __str__(self):
        """Return a string representation of the dataset."""
        return f"Features: {len(self.features)}, Labels: {len(self.labels)}"

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Return the feature and label at the given index."""
        return self.features.iloc[idx], self.labels.iloc[idx]

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i in range(len(self)):
            yield self[i]

    def save(self, features_path: str, labels_path: str):
        """Save the features and labels to CSV files."""
        self.features.to_csv(features_path, index=False)
        self.labels.to_csv(labels_path, index=False)

    @classmethod
    def load(cls, features_path: str, labels_path: str):
        """Load the features and labels from CSV files."""
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        return cls(features=features, labels=labels)
