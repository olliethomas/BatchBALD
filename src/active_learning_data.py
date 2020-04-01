from typing import List
import numpy as np
import torch.utils.data as data
import torch
from torch.utils.data import Subset
from torch import Tensor


class ActiveLearningData(object):
    """Splits `dataset` into an active dataset and an available dataset."""

    def __init__(self, dataset: data.Dataset):
        super().__init__()
        self.dataset = dataset
        self.active_mask = np.full((len(dataset),), False)
        self.available_mask = np.full((len(dataset),), True)

        self.active_dataset = data.Subset(self.dataset, [])
        self.available_dataset = data.Subset(self.dataset, [])

        self._update_indices()

    def _update_indices(self) -> None:
        self.active_dataset.indices = np.nonzero(self.active_mask)[0]  # type: ignore[assignment]
        self.available_dataset.indices = np.nonzero(self.available_mask)[0]  # type: ignore[assignment]

    def get_dataset_indices(self, available_indices: Tensor) -> Tensor:
        indices = self.available_dataset.indices[available_indices]  # type: ignore[call-overload]
        return indices

    def acquire(self, available_indices: Tensor) -> None:
        indices = self.get_dataset_indices(available_indices)

        self.active_mask[indices] = True
        self.available_mask[indices] = False
        self._update_indices()

    def make_unavailable(self, available_indices: Tensor) -> None:
        indices = self.get_dataset_indices(available_indices)

        self.available_mask[indices] = False
        self._update_indices()

    def get_random_available_indices(self, size: int) -> Tensor:
        assert 0 <= size <= len(self.available_dataset)
        available_indices = torch.randperm(len(self.available_dataset))[:size]
        return available_indices

    def extract_dataset(self, size: int) -> Subset:
        """Extract a dataset randomly from the available dataset and make those indices unavailable."""
        return self.extract_dataset_from_indices(self.get_random_available_indices(size))

    def extract_dataset_from_indices(self, available_indices: Tensor) -> Subset:
        """Extract a dataset from the available dataset and make those indices unavailable."""
        dataset_indices = self.get_dataset_indices(available_indices)

        self.make_unavailable(available_indices)
        return data.Subset(self.dataset, dataset_indices)  # type: ignore[arg-type]
