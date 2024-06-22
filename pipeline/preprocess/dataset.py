"""Define the TransactionDataset class."""
from dataclasses import dataclass
from typing import Any

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
        return self.features[idx], self.labels[idx]
    def __iter__(self):
        """Return an iterator over the dataset."""
        for i in range(len(self)):
            yield self[i]
